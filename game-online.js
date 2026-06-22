const MODEL_ASSET_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const VISION_MODULE_URL = "./vendor/mediapipe/tasks-vision/vision_bundle.mjs";
const VISION_WASM_ROOT = "./vendor/mediapipe/tasks-vision/wasm";

const video = document.getElementById("cameraVideo");
const poseCanvas = document.getElementById("poseCanvas");
const poseCtx = poseCanvas.getContext("2d");
const runnerCanvas = document.getElementById("runnerCanvas");
const runnerCtx = runnerCanvas.getContext("2d");
const startButton = document.getElementById("startButton");
const gameBanner = document.getElementById("gameBanner");
const bannerText = document.getElementById("bannerText");
const cameraStatus = document.getElementById("cameraStatus");
const poseStatus = document.getElementById("poseStatus");
const gameStatus = document.getElementById("gameStatus");
const scoreValue = document.getElementById("scoreValue");
const bestValue = document.getElementById("bestValue");
const speedValue = document.getElementById("speedValue");
const actionBadges = Array.from(document.querySelectorAll("[data-action]"));

const KEYPOINT = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
};

const ACTION_LABEL = {
  left: "Geser kiri",
  right: "Geser kanan",
  jump: "Lompat",
  slide: "Slide",
  idle: "Siaga",
};

let poseLandmarker = null;
let webcamStream = null;
let animationFrameId = 0;
let visionReady = false;
let gameStarted = false;
let lastVideoTime = -1;
let lastPoseTimestampMs = -1;
let baselineTorso = 0;
let activeBadgeTimer = 0;
let FilesetResolverLib = null;
let PoseLandmarkerLib = null;

const inputState = {
  lastAction: "idle",
  confidence: 0,
  cooldownUntil: {
    left: 0,
    right: 0,
    jump: 0,
    slide: 0,
  },
};

const gameState = {
  running: false,
  over: false,
  score: 0,
  best: Number(localStorage.getItem("pose-runner-best-score") || 0),
  speed: 360,
  lane: 1,
  targetLane: 1,
  playerX: 0,
  jumpUntil: 0,
  slideUntil: 0,
  invulnerableUntil: 0,
  distance: 0,
  lastFrameAt: 0,
  lastSpawnAt: 0,
  obstacles: [],
  stars: [],
  lastScoredObstacleId: -1,
  nextObstacleId: 1,
};

bestValue.textContent = String(gameState.best);
resizeCanvases();
seedStars();
drawGame(performance.now());
cameraStatus.textContent = "Engine browser siap. Tekan mulai untuk memuat model dan meminta izin kamera.";
updatePoseStatus("Menunggu aktivasi kamera.");
updateGameStatus("Siap bermain. Tombol akan memulai kamera dan pose engine.");

window.addEventListener("error", (event) => {
  const message = event.error?.message || event.message || "Terjadi error yang tidak dikenal.";
  cameraStatus.textContent = `Error runtime: ${message}`;
  bannerText.textContent =
    "Terjadi error saat menjalankan game di browser. Coba refresh, lalu tekan mulai lagi.";
  startButton.disabled = false;
  startButton.textContent = "Coba Lagi";
});

window.addEventListener("unhandledrejection", (event) => {
  const reason =
    event.reason instanceof Error ? event.reason.message : String(event.reason || "Promise gagal.");
  cameraStatus.textContent = `Promise gagal: ${reason}`;
  bannerText.textContent =
    "Browser gagal memuat modul pose atau akses kamera. Cek koneksi internet dan izin kamera.";
  startButton.disabled = false;
  startButton.textContent = "Coba Lagi";
});

window.addEventListener("resize", () => {
  resizeCanvases();
  seedStars();
});

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  startButton.textContent = "Menyiapkan kamera...";

  try {
    if (!visionReady) {
      await setupPoseLandmarker();
    }

    if (!webcamStream) {
      await setupCamera();
    }

    if (!gameStarted) {
      gameStarted = true;
      startGame();
      loop(performance.now());
    } else if (gameState.over) {
      restartGame();
    } else {
      resumeGame();
    }

    gameBanner.hidden = true;
    updateGameStatus("Kamera aktif. Gerakkan tubuhmu untuk menghindari rintangan.");
  } catch (error) {
    const message = error instanceof Error ? error.message : "Gagal mengaktifkan kamera.";
    cameraStatus.textContent = message;
    bannerText.textContent =
      "Browser gagal membuka kamera atau memuat model pose. Cek izin kamera, lalu coba lagi.";
    startButton.disabled = false;
    startButton.textContent = "Coba Lagi";
    updateGameStatus("Belum bisa mulai. Izin kamera atau load model gagal.");
  }
});

window.addEventListener("keydown", (event) => {
  if (!gameStarted) return;

  if (event.key === "ArrowLeft") {
    event.preventDefault();
    triggerAction("left", 1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    triggerAction("right", 1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    triggerAction("jump", 1);
  } else if (event.key === "ArrowDown") {
    event.preventDefault();
    triggerAction("slide", 1);
  } else if (event.key === " " && gameState.over) {
    event.preventDefault();
    restartGame();
  }
});

async function setupPoseLandmarker() {
  cameraStatus.textContent = "Mengunduh model pose...";

  await ensureVisionModule();

  let lastError = null;
  try {
    const vision = await FilesetResolverLib.forVisionTasks(VISION_WASM_ROOT);

    poseLandmarker = await PoseLandmarkerLib.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_ASSET_URL,
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.55,
      minPosePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
    });

    visionReady = true;
    cameraStatus.textContent = "Model pose siap. Meminta akses kamera...";
    return;
  } catch (error) {
    lastError = error;
  }

  throw new Error(
    `Gagal memuat MediaPipe WebAssembly lokal. ${lastError instanceof Error ? lastError.message : ""}`.trim()
  );
}

async function setupCamera() {
  if (!window.isSecureContext) {
    throw new Error("Kamera browser hanya bisa dipakai di context HTTPS atau localhost.");
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Browser ini tidak mendukung getUserMedia untuk akses kamera.");
  }

  webcamStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 960 },
      height: { ideal: 720 },
    },
    audio: false,
  });

  video.srcObject = webcamStream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play().then(resolve).catch(resolve);
    };
  });

  cameraStatus.textContent = "Kamera aktif. Menunggu pose...";
}

async function ensureVisionModule() {
  if (FilesetResolverLib && PoseLandmarkerLib) {
    return;
  }

  try {
    const visionModule = await import(VISION_MODULE_URL);
    FilesetResolverLib = visionModule.FilesetResolver;
    PoseLandmarkerLib = visionModule.PoseLandmarker;
  } catch (error) {
    throw new Error(
      `Gagal memuat modul MediaPipe lokal. ${error instanceof Error ? error.message : ""}`.trim()
    );
  }

  if (!FilesetResolverLib || !PoseLandmarkerLib) {
    throw new Error("Ekspor MediaPipe lokal tidak lengkap.");
  }
}

function resizeCanvases() {
  const runnerRect = runnerCanvas.getBoundingClientRect();
  const videoRect = video.getBoundingClientRect();

  if (runnerRect.width > 0 && runnerRect.height > 0) {
    runnerCanvas.width = Math.round(runnerRect.width * window.devicePixelRatio);
    runnerCanvas.height = Math.round(runnerRect.height * window.devicePixelRatio);
    runnerCtx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  }

  if (videoRect.width > 0 && videoRect.height > 0) {
    poseCanvas.width = Math.round(videoRect.width * window.devicePixelRatio);
    poseCanvas.height = Math.round(videoRect.height * window.devicePixelRatio);
    poseCtx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  }
}

function seedStars() {
  const width = runnerCanvas.getBoundingClientRect().width || 900;
  const height = runnerCanvas.getBoundingClientRect().height || 700;
  gameState.stars = Array.from({ length: 34 }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    size: Math.random() * 2.4 + 0.8,
    speed: Math.random() * 18 + 14,
    alpha: Math.random() * 0.45 + 0.25,
  }));
}

function startGame() {
  restartGame();
  startButton.textContent = "Main Lagi";
}

function resumeGame() {
  if (!gameState.over) {
    gameState.running = true;
    gameState.lastFrameAt = performance.now();
    updateGameStatus("Game berjalan. Pose aktif.");
  }
}

function restartGame() {
  gameState.running = true;
  gameState.over = false;
  lastVideoTime = -1;
  lastPoseTimestampMs = -1;
  gameState.score = 0;
  gameState.speed = 360;
  gameState.lane = 1;
  gameState.targetLane = 1;
  gameState.jumpUntil = 0;
  gameState.slideUntil = 0;
  gameState.invulnerableUntil = performance.now() + 500;
  gameState.distance = 0;
  gameState.lastFrameAt = performance.now();
  gameState.lastSpawnAt = performance.now();
  gameState.obstacles = [];
  gameState.lastScoredObstacleId = -1;
  updateScoreUI();
  gameBanner.hidden = true;
  cameraStatus.textContent = "Kamera aktif. Menunggu pose...";
  updatePoseStatus("Mode aktif. Angkat tangan atau jongkok untuk mengontrol.");
  updateGameStatus("Game dimulai. Jaga ritme gerakanmu.");
}

function endGame() {
  gameState.running = false;
  gameState.over = true;
  gameState.best = Math.max(gameState.best, gameState.score);
  localStorage.setItem("pose-runner-best-score", String(gameState.best));
  bestValue.textContent = String(gameState.best);
  gameBanner.hidden = false;
  bannerText.textContent =
    "Karakter menabrak rintangan. Tekan Main Lagi atau tombol spasi untuk restart.";
  startButton.disabled = false;
  startButton.textContent = "Main Lagi";
  updateGameStatus("Game over. Kamu bisa langsung restart.");
}

function loop(timestamp) {
  const deltaMs = Math.min(32, timestamp - (gameState.lastFrameAt || timestamp));
  const deltaSec = deltaMs / 1000;
  gameState.lastFrameAt = timestamp;

  processPoseFrame(timestamp);

  if (gameState.running) {
    updateGame(deltaSec, timestamp);
  }

  drawGame(timestamp);
  animationFrameId = window.requestAnimationFrame(loop);
}

function processPoseFrame(timestamp) {
  if (!poseLandmarker || !video.videoWidth || video.readyState < 2) return;
  if (lastVideoTime === video.currentTime) return;

  lastVideoTime = video.currentTime;
  const mediaTimestampMs = Math.round(video.currentTime * 1000);
  const safeTimestampMs = Math.max(lastPoseTimestampMs + 1, mediaTimestampMs, Math.round(timestamp));
  lastPoseTimestampMs = safeTimestampMs;

  let result;
  try {
    result = poseLandmarker.detectForVideo(video, safeTimestampMs);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    cameraStatus.textContent = `Pose engine error: ${message}`;
    updatePoseStatus("Frame pose dilewati karena timestamp tidak valid. Sistem akan mencoba lanjut.");
    return;
  }

  drawPose(result);

  const landmarks = result.landmarks?.[0];
  if (!landmarks) {
    updatePoseStatus("Pose tidak terlihat jelas. Pastikan seluruh badan bagian atas masuk frame.");
    setActiveBadge("idle");
    return;
  }

  const poseDecision = inferPoseAction(landmarks);
  updatePoseStatus(
    `${ACTION_LABEL[poseDecision.action]} | confidence ${Math.round(poseDecision.confidence * 100)}%`
  );

  if (poseDecision.action !== "idle") {
    triggerAction(poseDecision.action, poseDecision.confidence);
  } else {
    setActiveBadge("idle");
  }
}

function inferPoseAction(landmarks) {
  const leftShoulder = landmarks[KEYPOINT.leftShoulder];
  const rightShoulder = landmarks[KEYPOINT.rightShoulder];
  const leftWrist = landmarks[KEYPOINT.leftWrist];
  const rightWrist = landmarks[KEYPOINT.rightWrist];
  const leftHip = landmarks[KEYPOINT.leftHip];
  const rightHip = landmarks[KEYPOINT.rightHip];

  const required = [leftShoulder, rightShoulder, leftWrist, rightWrist, leftHip, rightHip];
  if (required.some((point) => !point)) {
    return { action: "idle", confidence: 0 };
  }

  const visibleConfidence =
    required.reduce((sum, point) => sum + (point.visibility ?? 0.8), 0) / required.length;

  const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
  const hipY = (leftHip.y + rightHip.y) / 2;
  const torsoHeight = Math.max(hipY - shoulderY, 0.001);

  if (baselineTorso === 0) {
    baselineTorso = torsoHeight;
  } else if (torsoHeight > baselineTorso * 0.8) {
    baselineTorso = baselineTorso * 0.92 + torsoHeight * 0.08;
  }

  const leftRaised = leftWrist.y < leftShoulder.y - torsoHeight * 0.18;
  const rightRaised = rightWrist.y < rightShoulder.y - torsoHeight * 0.18;
  const crouchRatio = torsoHeight / Math.max(baselineTorso, 0.001);
  const isCrouching = crouchRatio < 0.72;

  const bothHandsScore = clamp(
    ((leftShoulder.y - leftWrist.y) + (rightShoulder.y - rightWrist.y)) / (torsoHeight * 2.2),
    0,
    1
  );
  const leftScore = clamp((leftShoulder.y - leftWrist.y) / (torsoHeight * 1.3), 0, 1);
  const rightScore = clamp((rightShoulder.y - rightWrist.y) / (torsoHeight * 1.3), 0, 1);
  const crouchScore = clamp((0.82 - crouchRatio) / 0.24, 0, 1);

  if (visibleConfidence < 0.45) {
    return { action: "idle", confidence: visibleConfidence };
  }

  if (leftRaised && rightRaised && bothHandsScore > 0.45) {
    return { action: "jump", confidence: Math.min(1, bothHandsScore * visibleConfidence) };
  }

  if (isCrouching && crouchScore > 0.3) {
    return { action: "slide", confidence: Math.min(1, crouchScore * visibleConfidence) };
  }

  if (leftRaised && !rightRaised && leftScore > 0.42) {
    return { action: "left", confidence: Math.min(1, leftScore * visibleConfidence) };
  }

  if (rightRaised && !leftRaised && rightScore > 0.42) {
    return { action: "right", confidence: Math.min(1, rightScore * visibleConfidence) };
  }

  return { action: "idle", confidence: visibleConfidence * 0.5 };
}

function triggerAction(action, confidence) {
  const now = performance.now();
  const cooldownMs = getCooldown(action);

  if (now < inputState.cooldownUntil[action]) return;

  inputState.lastAction = action;
  inputState.confidence = confidence;
  inputState.cooldownUntil[action] = now + cooldownMs;

  if (action === "left") {
    gameState.targetLane = Math.max(0, gameState.targetLane - 1);
  } else if (action === "right") {
    gameState.targetLane = Math.min(2, gameState.targetLane + 1);
  } else if (action === "jump") {
    if (now > gameState.jumpUntil - 160) {
      gameState.jumpUntil = now + 780;
    }
  } else if (action === "slide") {
    if (now > gameState.slideUntil - 160) {
      gameState.slideUntil = now + 760;
    }
  }

  setActiveBadge(action);
}

function getCooldown(action) {
  if (action === "left" || action === "right") return 420;
  if (action === "jump") return 900;
  if (action === "slide") return 880;
  return 250;
}

function setActiveBadge(action) {
  if (action === "idle") {
    if (performance.now() < activeBadgeTimer) return;
    actionBadges.forEach((badge) => badge.classList.remove("active"));
    return;
  }

  activeBadgeTimer = performance.now() + 260;
  actionBadges.forEach((badge) => {
    badge.classList.toggle("active", badge.dataset.action === action);
  });
}

function updateGame(deltaSec, timestamp) {
  const width = runnerCanvas.getBoundingClientRect().width || 960;
  const laneXs = [width * 0.3, width * 0.5, width * 0.7];
  gameState.lane += (gameState.targetLane - gameState.lane) * Math.min(1, deltaSec * 11);
  gameState.playerX = lerpLane(laneXs, gameState.lane);
  gameState.speed = Math.min(820, gameState.speed + deltaSec * 11);
  gameState.distance += deltaSec * gameState.speed;

  const spawnEveryMs = Math.max(560, 1120 - gameState.speed * 0.6);
  if (timestamp - gameState.lastSpawnAt > spawnEveryMs) {
    spawnObstacle();
    gameState.lastSpawnAt = timestamp;
  }

  for (const star of gameState.stars) {
    star.y += deltaSec * star.speed;
    if (star.y > (runnerCanvas.getBoundingClientRect().height || 720) + 5) {
      star.y = -5;
      star.x = Math.random() * width;
    }
  }

  const obstacleSpeed = deltaSec * gameState.speed;
  const playerY = (runnerCanvas.getBoundingClientRect().height || 720) - 138;
  const jumpProgress = getJumpProgress(timestamp);
  const isSliding = timestamp < gameState.slideUntil;
  const playerHitboxY = playerY - jumpProgress * 118;

  gameState.obstacles = gameState.obstacles.filter((obstacle) => {
    obstacle.y += obstacleSpeed;

    if (obstacle.y > (runnerCanvas.getBoundingClientRect().height || 720) + 120) {
      if (obstacle.id > gameState.lastScoredObstacleId) {
        gameState.score += 1;
        gameState.lastScoredObstacleId = obstacle.id;
        updateScoreUI();
      }
      return false;
    }

    const sameLane = Math.abs(obstacle.lane - gameState.lane) < 0.28;
    const inHitWindow =
      obstacle.y > playerHitboxY - 56 &&
      obstacle.y < playerHitboxY + 46 &&
      timestamp > gameState.invulnerableUntil;

    if (!sameLane || !inHitWindow) return true;

    const clearedByJump = obstacle.type === "barrier" && jumpProgress > 0.36;
    const clearedBySlide = obstacle.type === "gate" && isSliding;

    if (!clearedByJump && !clearedBySlide) {
      endGame();
    }

    return true;
  });

  speedValue.textContent = `${(gameState.speed / 360).toFixed(1)}x`;
}

function spawnObstacle() {
  const lanes = [0, 1, 2];
  const lane = lanes[Math.floor(Math.random() * lanes.length)];
  const type = Math.random() > 0.42 ? "barrier" : "gate";

  gameState.obstacles.push({
    id: gameState.nextObstacleId++,
    lane,
    type,
    y: -140,
  });
}

function drawGame(timestamp) {
  const width = runnerCanvas.getBoundingClientRect().width || 960;
  const height = runnerCanvas.getBoundingClientRect().height || 720;

  runnerCtx.clearRect(0, 0, width, height);

  const sky = runnerCtx.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, "#09111f");
  sky.addColorStop(0.55, "#0f1d34");
  sky.addColorStop(1, "#050913");
  runnerCtx.fillStyle = sky;
  runnerCtx.fillRect(0, 0, width, height);

  drawStars(width, height);
  drawCity(width, height);
  drawRoad(width, height, timestamp);
  drawObstacles(width, height);
  drawPlayer(width, height, timestamp);
}

function drawStars(width, height) {
  for (const star of gameState.stars) {
    runnerCtx.fillStyle = `rgba(180, 221, 255, ${star.alpha})`;
    runnerCtx.beginPath();
    runnerCtx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
    runnerCtx.fill();
  }
}

function drawCity(width, height) {
  const baseY = height * 0.36;
  for (let i = 0; i < 9; i += 1) {
    const x = i * (width / 8.5);
    const towerWidth = 42 + ((i * 17) % 40);
    const towerHeight = 130 + ((i * 33) % 140);
    runnerCtx.fillStyle = "rgba(7, 13, 24, 0.78)";
    runnerCtx.fillRect(x, baseY - towerHeight, towerWidth, towerHeight);

    for (let row = 0; row < 5; row += 1) {
      for (let col = 0; col < 2; col += 1) {
        runnerCtx.fillStyle = row % 2 === 0 ? "rgba(247, 179, 43, 0.25)" : "rgba(42, 209, 255, 0.18)";
        runnerCtx.fillRect(x + 8 + col * 14, baseY - towerHeight + 12 + row * 18, 6, 10);
      }
    }
  }
}

function drawRoad(width, height, timestamp) {
  const roadTopWidth = width * 0.18;
  const roadBottomWidth = width * 0.78;
  const centerX = width / 2;
  const topY = height * 0.22;
  const bottomY = height;
  const leftTop = centerX - roadTopWidth / 2;
  const rightTop = centerX + roadTopWidth / 2;
  const leftBottom = centerX - roadBottomWidth / 2;
  const rightBottom = centerX + roadBottomWidth / 2;

  runnerCtx.fillStyle = "#0b1323";
  runnerCtx.beginPath();
  runnerCtx.moveTo(leftTop, topY);
  runnerCtx.lineTo(rightTop, topY);
  runnerCtx.lineTo(rightBottom, bottomY);
  runnerCtx.lineTo(leftBottom, bottomY);
  runnerCtx.closePath();
  runnerCtx.fill();

  runnerCtx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  runnerCtx.lineWidth = 2;

  for (const laneFactor of [1 / 3, 2 / 3]) {
    const topX = leftTop + (rightTop - leftTop) * laneFactor;
    const bottomX = leftBottom + (rightBottom - leftBottom) * laneFactor;
    runnerCtx.beginPath();
    runnerCtx.moveTo(topX, topY);
    runnerCtx.lineTo(bottomX, bottomY);
    runnerCtx.stroke();
  }

  const dashOffset = (timestamp * 0.42) % 100;
  for (let i = 0; i < 12; i += 1) {
    const t = (i * 85 + dashOffset) / (height + 220);
    const y = topY + t * (bottomY - topY);
    const laneCenterTop = centerX;
    const laneCenterBottom = centerX;
    const x = laneCenterTop + (laneCenterBottom - laneCenterTop) * t;
    const dashWidth = 6 + t * 20;
    const dashHeight = 18 + t * 42;

    runnerCtx.fillStyle = "rgba(255, 240, 180, 0.58)";
    runnerCtx.fillRect(x - dashWidth / 2, y, dashWidth, dashHeight);
  }

  const neon = runnerCtx.createLinearGradient(0, topY, 0, bottomY);
  neon.addColorStop(0, "rgba(42, 209, 255, 0)");
  neon.addColorStop(0.5, "rgba(42, 209, 255, 0.18)");
  neon.addColorStop(1, "rgba(42, 209, 255, 0)");
  runnerCtx.strokeStyle = neon;
  runnerCtx.lineWidth = 8;
  runnerCtx.beginPath();
  runnerCtx.moveTo(leftBottom + 6, bottomY);
  runnerCtx.lineTo(leftTop, topY);
  runnerCtx.moveTo(rightBottom - 6, bottomY);
  runnerCtx.lineTo(rightTop, topY);
  runnerCtx.stroke();
}

function drawObstacles(width, height) {
  const laneXs = [width * 0.3, width * 0.5, width * 0.7];

  for (const obstacle of gameState.obstacles) {
    const perspective = clamp(obstacle.y / height, 0, 1);
    const x = laneXs[obstacle.lane];
    const scale = 0.55 + perspective * 0.9;

    if (obstacle.type === "barrier") {
      const w = 46 * scale;
      const h = 40 * scale;
      runnerCtx.fillStyle = "#ff6f59";
      runnerCtx.fillRect(x - w / 2, obstacle.y - h / 2, w, h);
      runnerCtx.fillStyle = "#ffd978";
      runnerCtx.fillRect(x - w / 2, obstacle.y - h / 2, w, 8 * scale);
    } else {
      const w = 72 * scale;
      const h = 96 * scale;
      runnerCtx.strokeStyle = "#6fffb0";
      runnerCtx.lineWidth = 8 * scale;
      runnerCtx.strokeRect(x - w / 2, obstacle.y - h / 2, w, h);
      runnerCtx.fillStyle = "rgba(111, 255, 176, 0.14)";
      runnerCtx.fillRect(x - w / 2, obstacle.y - h / 2, w, 16 * scale);
    }
  }
}

function drawPlayer(width, height, timestamp) {
  const laneXs = [width * 0.3, width * 0.5, width * 0.7];
  const x = lerpLane(laneXs, gameState.lane);
  const baseY = height - 138;
  const jumpProgress = getJumpProgress(timestamp);
  const slideProgress = timestamp < gameState.slideUntil ? 1 : 0;
  const y = baseY - jumpProgress * 118;
  const bodyHeight = slideProgress ? 42 : 78;
  const bodyWidth = slideProgress ? 64 : 44;

  runnerCtx.save();
  runnerCtx.translate(x, y);

  runnerCtx.fillStyle = "rgba(42, 209, 255, 0.18)";
  runnerCtx.beginPath();
  runnerCtx.ellipse(0, 48, slideProgress ? 52 : 38, 16, 0, 0, Math.PI * 2);
  runnerCtx.fill();

  runnerCtx.fillStyle = "#f7b32b";
  runnerCtx.beginPath();
  runnerCtx.arc(0, -bodyHeight * 0.72, 16, 0, Math.PI * 2);
  runnerCtx.fill();

  runnerCtx.fillStyle = "#6fffb0";
  runnerCtx.fillRect(-bodyWidth / 2, -bodyHeight / 2, bodyWidth, bodyHeight);

  runnerCtx.strokeStyle = "#08111f";
  runnerCtx.lineWidth = 8;
  runnerCtx.beginPath();
  runnerCtx.moveTo(-10, bodyHeight / 2 - 2);
  runnerCtx.lineTo(-16, bodyHeight / 2 + 28);
  runnerCtx.moveTo(10, bodyHeight / 2 - 2);
  runnerCtx.lineTo(16, bodyHeight / 2 + 28);
  runnerCtx.moveTo(-bodyWidth / 2, -8);
  runnerCtx.lineTo(-34, 14);
  runnerCtx.moveTo(bodyWidth / 2, -8);
  runnerCtx.lineTo(34, 14);
  runnerCtx.stroke();

  runnerCtx.restore();
}

function drawPose(result) {
  const width = video.getBoundingClientRect().width || 640;
  const height = video.getBoundingClientRect().height || 480;
  poseCtx.clearRect(0, 0, width, height);

  const landmarks = result.landmarks?.[0];
  if (!landmarks) return;

  const segments = [
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [11, 23],
    [12, 24],
    [23, 24],
  ];

  poseCtx.lineWidth = 3;
  poseCtx.strokeStyle = "rgba(111, 255, 176, 0.9)";
  poseCtx.fillStyle = "rgba(247, 179, 43, 0.95)";

  for (const [fromIndex, toIndex] of segments) {
    const from = landmarks[fromIndex];
    const to = landmarks[toIndex];
    if (!from || !to) continue;

    poseCtx.beginPath();
    poseCtx.moveTo(from.x * width, from.y * height);
    poseCtx.lineTo(to.x * width, to.y * height);
    poseCtx.stroke();
  }

  for (const index of Object.values(KEYPOINT)) {
    const point = landmarks[index];
    if (!point) continue;

    poseCtx.beginPath();
    poseCtx.arc(point.x * width, point.y * height, 6, 0, Math.PI * 2);
    poseCtx.fill();
  }
}

function updateScoreUI() {
  scoreValue.textContent = String(gameState.score);
}

function updatePoseStatus(message) {
  poseStatus.textContent = message;
}

function updateGameStatus(message) {
  gameStatus.textContent = message;
}

function getJumpProgress(timestamp) {
  if (timestamp > gameState.jumpUntil) return 0;
  const remaining = gameState.jumpUntil - timestamp;
  const progress = 1 - remaining / 780;
  return Math.sin(progress * Math.PI);
}

function lerpLane(lanes, laneValue) {
  const leftIndex = Math.floor(laneValue);
  const rightIndex = Math.min(lanes.length - 1, Math.ceil(laneValue));
  const amount = laneValue - leftIndex;
  return lanes[leftIndex] + (lanes[rightIndex] - lanes[leftIndex]) * amount;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

window.addEventListener("beforeunload", () => {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }

  webcamStream?.getTracks().forEach((track) => track.stop());
});
