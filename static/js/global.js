document.addEventListener("DOMContentLoaded", () => {
  // Attach showLoading to all POST/GET forms
  document.querySelectorAll("form[method='post'], form[method='get']").forEach((form) => {
    form.addEventListener("submit", () => {
      const spinner = document.getElementById("loadingSpinner");
      const timerLabel = document.getElementById("loadingTimer");

      if (spinner) spinner.style.display = "flex";

      let seconds = 0;
      window.loadingInterval = setInterval(() => {
        seconds++;
        if (timerLabel) timerLabel.textContent = `⏳ ${seconds}s`;
      }, 1000);
    });
  });

  // Optional: Show Password toggle
  const toggleBtn = document.getElementById("togglePassword");
  const pwdField = document.getElementById("password");

  if (toggleBtn && pwdField) {
    toggleBtn.addEventListener("click", () => {
      const type = pwdField.getAttribute("type") === "password" ? "text" : "password";
      pwdField.setAttribute("type", type);
      toggleBtn.textContent = type === "password" ? "Show" : "Hide";
    });
  }
});
function startSpinner() {
  const spinner = document.getElementById('loadingSpinner');
  const timer = document.getElementById('loadingTimer');
  if (!spinner || !timer) return;

  spinner.style.display = 'flex';
  let seconds = 0;
  timer.textContent = '⏳ 0s';
  if (window.loadingInterval) clearInterval(window.loadingInterval);
  window.loadingInterval = setInterval(() => {
    seconds++;
    timer.textContent = `⏳ ${seconds}s`;
  }, 1000);
}