(() => {
  "use strict";

  const data = window.ERYNDOR_DATA;
  if (!data) {
    document.body.innerHTML = "<p>No se pudieron cargar las métricas de la presentación.</p>";
    return;
  }

  const slides = Array.from(document.querySelectorAll(".slide"));
  const chapterList = document.querySelector("#chapter-list");
  const counter = document.querySelector("#slide-counter");
  const progress = document.querySelector("#progress");
  let current = 0;

  const readPath = (object, path) => path.split(".").reduce((value, key) => value?.[key], object);

  document.querySelectorAll("[data-bind]").forEach((element) => {
    const value = readPath(data, element.dataset.bind);
    if (value !== undefined && value !== null) element.textContent = value;
  });

  document.querySelector("#members").textContent = data.members.join(" · ");
  document.querySelector("#gallery-cover-label").textContent = data.gallery_ready ? "Galería validada" : "Meta de galería";

  const epochStatus = Object.entries(data.completed_epochs)
    .map(([name, epoch]) => `${name.replaceAll("_", " ")}: ${epoch}/${data.target_epochs}`)
    .join(" · ");
  document.querySelector("#epoch-status").textContent = epochStatus;
  document.querySelector("#matrix-training-limit").textContent = data.training_complete
    ? "Trayectorias completas."
    : `Estado ${data.minimum_epoch}/${data.target_epochs}; sin ganador.`;

  const metricHost = document.querySelector("#experiment-metrics");
  data.experiments.forEach((experiment) => {
    const row = document.createElement("div");
    [experiment.label, `D ${experiment.loss_d}`, `G ${experiment.loss_g}`, `div ${experiment.diversity}`]
      .forEach((value, index) => {
        const cell = document.createElement(index === 0 ? "b" : "span");
        cell.textContent = value;
        row.appendChild(cell);
      });
    metricHost.appendChild(row);
  });
  document.querySelector("#training-state").textContent = data.training_complete
    ? `Fase 8 completa · modelo provisional: ${data.selected_experiment}.`
    : "Resultado parcial: la comparación final sigue bloqueada.";

  if (data.phase8_complete && !data.gallery_ready) {
    document.querySelector("#learning-limit").textContent =
      "A y C aprendieron personajes; todavía falta demostrar novedad frente al entrenamiento.";
  }

  if (data.gallery_ready) {
    document.querySelectorAll("[data-src]").forEach((image) => { image.src = image.dataset.src; });
    document.querySelector("#gallery-pending").hidden = true;
    document.querySelector("#gallery-ready").hidden = false;
    document.querySelector("#neighbors-pending").hidden = true;
    document.querySelector("#neighbors-ready").hidden = false;
    document.querySelector("#learning-limit").textContent = "La galería validada se regenera con diferencia RGB máxima igual a cero.";
    document.querySelector("#reflection").innerHTML = `<strong>${data.hardest_character}</strong> fue el caso más difícil de defender como nuevo (coseno ${data.hardest_similarity}).`;
    document.querySelector("#matrix-gallery").textContent = "Galería validada y tasa declarada.";
    document.querySelector("#matrix-gallery-limit").textContent = "Selección humana inevitable.";
    document.querySelector("#matrix-neighbors").textContent = "Novedad medida con coseno y MSE.";
  }

  slides.forEach((slide, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.title = `${index + 1}. ${slide.dataset.title}`;
    button.setAttribute("aria-label", `Ir a ${slide.dataset.title}`);
    button.addEventListener("click", () => show(index));
    item.appendChild(button);
    chapterList.appendChild(item);
    slide.addEventListener("click", () => {
      if (document.body.classList.contains("overview")) {
        document.body.classList.remove("overview");
        show(index);
      }
    });
  });

  const railButtons = Array.from(chapterList.querySelectorAll("button"));

  function show(index, updateHash = true) {
    current = Math.max(0, Math.min(index, slides.length - 1));
    slides.forEach((slide, slideIndex) => {
      const active = slideIndex === current;
      slide.classList.toggle("is-active", active);
      slide.setAttribute("aria-hidden", String(!active));
      railButtons[slideIndex].setAttribute("aria-current", String(active));
    });
    counter.textContent = `${current + 1} / ${slides.length}`;
    progress.style.width = `${((current + 1) / slides.length) * 100}%`;
    if (updateHash) history.replaceState(null, "", `#slide-${current + 1}`);
  }

  function toggleOverview() {
    document.body.classList.toggle("overview");
    if (!document.body.classList.contains("overview")) show(current);
  }

  async function toggleFullscreen() {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  }

  document.querySelector("#prev-button").addEventListener("click", () => show(current - 1));
  document.querySelector("#next-button").addEventListener("click", () => show(current + 1));
  document.querySelector("#overview-button").addEventListener("click", toggleOverview);
  document.querySelector("#fullscreen-button").addEventListener("click", toggleFullscreen);

  window.addEventListener("keydown", (event) => {
    if (["ArrowRight", "PageDown", " "].includes(event.key)) { event.preventDefault(); show(current + 1); }
    if (["ArrowLeft", "PageUp"].includes(event.key)) { event.preventDefault(); show(current - 1); }
    if (event.key === "Home") { event.preventDefault(); show(0); }
    if (event.key === "End") { event.preventDefault(); show(slides.length - 1); }
    if (event.key.toLowerCase() === "o") toggleOverview();
    if (event.key.toLowerCase() === "f") toggleFullscreen();
    if (event.key === "Escape" && document.body.classList.contains("overview")) toggleOverview();
  });

  const initial = Number.parseInt(location.hash.replace("#slide-", ""), 10);
  show(Number.isFinite(initial) ? initial - 1 : 0, false);
})();
