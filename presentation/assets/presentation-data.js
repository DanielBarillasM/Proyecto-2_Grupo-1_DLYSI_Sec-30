window.ERYNDOR_DATA = {
  "project": "Eryndor: Guardianes del Velo",
  "members": [
    "Pablo Daniel Barillas Moreno",
    "Wilson Alejandro Calderón"
  ],
  "target_epochs": 60,
  "training_complete": false,
  "completed_epochs": {
    "baseline_bce": 1,
    "hinge_loss": 1,
    "bce_spectral_norm": 1
  },
  "minimum_epoch": 1,
  "dataset": {
    "images": 4096,
    "resolution": "64 × 64",
    "duplicates": 0,
    "missing": 0,
    "hair_styles": 44,
    "torso_styles": 18,
    "foreground": "23.18%",
    "credited_layers": "254 / 254"
  },
  "experiments": [
    {
      "id": "baseline_bce",
      "label": "A · BCE base",
      "epochs": 1,
      "loss_d": 0.1156,
      "loss_g": 8.5477,
      "diversity": 0.0551,
      "seconds": 184.4
    },
    {
      "id": "hinge_loss",
      "label": "B · Hinge",
      "epochs": 1,
      "loss_d": 0.2482,
      "loss_g": 15.2575,
      "diversity": 0.086,
      "seconds": 204.6
    },
    {
      "id": "bce_spectral_norm",
      "label": "C · BCE + spectral norm",
      "epochs": 1,
      "loss_d": 0.076,
      "loss_g": 6.8625,
      "diversity": 0.0618,
      "seconds": 185.2
    }
  ],
  "gallery_ready": false,
  "selection": "10 / 200 = 5%",
  "gallery": [],
  "mean_neighbor_similarity": null,
  "hardest_character": null,
  "hardest_similarity": null,
  "regeneration_delta": null
};
