window.ERYNDOR_DATA = {
  "project": "Eryndor: Guardianes del Velo",
  "members": [
    "Pablo Daniel Barillas Moreno",
    "Wilson Alejandro Calderón"
  ],
  "target_epochs": 60,
  "training_complete": true,
  "completed_epochs": {
    "baseline_bce": 60,
    "hinge_loss": 60,
    "bce_spectral_norm": 60
  },
  "minimum_epoch": 60,
  "phase8_complete": true,
  "selected_experiment": "bce_spectral_norm",
  "selection_scope": "provisional para fases 9–10; faltan vecinos y 200 candidatos",
  "phase9_ready": false,
  "phase9_checkpoint_epoch": 1,
  "phase9_target_epoch": 60,
  "phase9_resnet_cached": true,
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
      "epochs": 60,
      "loss_d": 0.0789,
      "loss_g": 5.6369,
      "diversity": 0.2585,
      "seconds": 6.1
    },
    {
      "id": "hinge_loss",
      "label": "B · Hinge",
      "epochs": 60,
      "loss_d": 0.0,
      "loss_g": 8.1271,
      "diversity": 0.0136,
      "seconds": 6.0
    },
    {
      "id": "bce_spectral_norm",
      "label": "C · BCE + spectral norm",
      "epochs": 60,
      "loss_d": 0.1905,
      "loss_g": 7.0557,
      "diversity": 0.2659,
      "seconds": 6.4
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
