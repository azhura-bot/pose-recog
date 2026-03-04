const revealItems = document.querySelectorAll(".reveal");

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("show");
      observer.unobserve(entry.target);
    });
  },
  { threshold: 0.15 }
);

revealItems.forEach((el, idx) => {
  el.style.setProperty("--reveal-delay", `${(idx % 4) * 90}ms`);
  observer.observe(el);
});

const meterFills = document.querySelectorAll(".meter-fill");

if (meterFills.length > 0) {
  const metersObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;

        meterFills.forEach((bar, idx) => {
          const level = Number(bar.dataset.level) || 0;
          setTimeout(() => {
            bar.style.width = `${Math.min(level, 100)}%`;
          }, idx * 110);
        });

        metersObserver.disconnect();
      });
    },
    { threshold: 0.4 }
  );

  metersObserver.observe(document.querySelector(".hero-visual"));
}

const cursorGlow = document.querySelector(".cursor-glow");
const canTrackCursor = window.matchMedia("(pointer: fine)").matches;

if (cursorGlow && canTrackCursor) {
  let currentX = window.innerWidth / 2;
  let currentY = window.innerHeight / 2;
  let targetX = currentX;
  let targetY = currentY;

  window.addEventListener("mousemove", (event) => {
    targetX = event.clientX;
    targetY = event.clientY;
    cursorGlow.style.opacity = "1";
  });

  window.addEventListener("mouseleave", () => {
    cursorGlow.style.opacity = "0";
  });

  const renderGlow = () => {
    currentX += (targetX - currentX) * 0.14;
    currentY += (targetY - currentY) * 0.14;
    cursorGlow.style.transform = `translate(${currentX - 110}px, ${currentY - 110}px)`;
    window.requestAnimationFrame(renderGlow);
  };

  renderGlow();
} else if (cursorGlow) {
  cursorGlow.remove();
}
