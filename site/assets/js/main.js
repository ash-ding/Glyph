/* Glyph project page — small progressive enhancements. No dependencies. */

(function () {
  "use strict";

  /* ---- copy buttons ------------------------------------------------------ */

  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest(".copy-btn");
    if (!btn) return;
    var target = document.getElementById(btn.dataset.copyTarget);
    if (!target || !navigator.clipboard) return;
    navigator.clipboard.writeText(target.innerText.trim()).then(function () {
      var was = btn.textContent;
      btn.textContent = "copied";
      setTimeout(function () { btn.textContent = was; }, 1400);
    });
  });

  /* ---- highlight the nav link for the section in view -------------------- */

  var links = Array.prototype.slice.call(
    document.querySelectorAll('.nav__links a[href^="#"]')
  );
  var sections = links
    .map(function (a) { return document.querySelector(a.getAttribute("href")); })
    .filter(Boolean);

  if (sections.length && "IntersectionObserver" in window) {
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        links.forEach(function (a) {
          a.toggleAttribute(
            "aria-current",
            a.getAttribute("href") === "#" + entry.target.id
          );
        });
      });
    }, { rootMargin: "-30% 0px -60% 0px" });
    sections.forEach(function (s) { obs.observe(s); });
  }
})();
