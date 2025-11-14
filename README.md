# 🎬 ComfyUI x Autodesk Flame - Ultimate VFX Toolkit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.5%2B-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Flame](https://img.shields.io/badge/Autodesk%20Flame-2024%2B-red.svg)](https://www.autodesk.com/products/flame)

> **L'outil ultime pour les graphistes** - Intégration complète entre Autodesk Flame et ComfyUI pour le traitement d'images et vidéos par IA.

---

## 🚀 Vue d'ensemble

Ce projet fournit une **intégration complète** entre **Autodesk Flame** et **ComfyUI**, permettant aux artistes VFX et graphistes d'utiliser la puissance de l'IA générative directement dans leur workflow de production.

### ✨ Fonctionnalités principales

- 🔌 **Communication WebSocket/REST** avec ComfyUI
- 🎨 **PyBox Handler** natif pour Autodesk Flame
- 🖼️ **Interface GUI** complète avec PyQt6
- 🎬 **Magic Wand Tool** pour la sélection et le masquage avancés
- 🚀 **6+ Workflows professionnels** prêts à l'emploi
- ⚡ **Traitement temps réel** avec suivi de progression
- 🎯 **Export/Import de workflows** ComfyUI

---

## 📁 Structure du projet

```
chatgpt/
│
├── 🎨 Core AI Integration
│   ├── comfyui_client.py                   # Client WebSocket/REST pour ComfyUI
│   ├── flame_comfyui_pybox.py              # Handler PyBox pour Autodesk Flame
│   └── comfyui_workflow_manager.py         # GUI gestionnaire de workflows
│
├── 🎬 Magic Wand Application
│   ├── greg_wand.py                        # Application Magic Wand principale
│   └── greg_wand_comfyui_integration.py    # Intégration ComfyUI dans Magic Wand
│
├── 🔧 Workflows professionnels
│   └── workflows/
│       ├── upscale_4k_professional.json            # Upscaling 4K/8K
│       ├── denoise_professional.json               # Denoising avancé
│       ├── controlnet_vfx_composition.json         # VFX avec ControlNet
│       ├── video_frame_interpolation.json          # Interpolation de frames
│       ├── color_grade_enhancement.json            # Color grading cinématique
│       └── texture_detail_enhancement.json         # Amélioration de détails
│
├── 📚 Documentation & Exemples
│   ├── README.md                           # Ce fichier
│   ├── README_COMFYUI_INTEGRATION.md      # Documentation détaillée
│   ├── example_usage.py                   # Exemples d'utilisation
│   └── install_dependencies.sh            # Script d'installation
│
└── ⚙️ Configuration
    ├── comfyui_config.json                # Configuration principale
    └── requirements.txt                   # Dépendances Python
```

---

## 🎯 Cas d'usage

### Pour les artistes VFX

✅ **Upscaling 4K/8K** - Montée en résolution avec raffinement IA
✅ **Denoising** - Suppression du bruit et artefacts
✅ **VFX Composition** - Composition avancée avec ControlNet
✅ **Frame Interpolation** - Génération de slow motion fluide

### Pour les graphistes

✅ **Color Grading** - Amélioration cinématique des couleurs
✅ **Texture Enhancement** - Amélioration des détails et textures
✅ **Style Transfer** - Application de styles artistiques
✅ **Restoration** - Restauration d'images anciennes

### Pour les studios

✅ **Workflow automatisé** - Traitement batch de clips Flame
✅ **Pipeline intégré** - Connexion directe Flame ↔ ComfyUI
✅ **Personnalisable** - Créez vos propres workflows
✅ **Production-ready** - Optimisé pour la production professionnelle

---

## 📦 Installation

### Prérequis

- Python 3.8 ou supérieur
- ComfyUI installé et fonctionnel
- (Optionnel) Autodesk Flame 2024+

### Installation rapide

```bash
# 1. Cloner le repository
git clone https://github.com/yourusername/chatgpt.git
cd chatgpt

# 2. Installer les dépendances
chmod +x install_dependencies.sh
./install_dependencies.sh

# 3. Lancer ComfyUI (dans un terminal séparé)
cd /path/to/ComfyUI
python main.py

# 4. Lancer l'interface
python3 comfyui_workflow_manager.py
```

### Installation manuelle

```bash
# Installer les dépendances Python
pip3 install -r requirements.txt

# Créer les répertoires
mkdir -p workflows /tmp/flame_comfyui
```

---

## 🚀 Démarrage rapide

### Mode 1 : Interface GUI standalone

```bash
# Lancer le gestionnaire de workflows
python3 comfyui_workflow_manager.py
```

**Workflow typique :**
1. Vérifiez la connexion à ComfyUI (🟢)
2. Sélectionnez un workflow dans la bibliothèque
3. Chargez une image d'entrée
4. Ajustez les paramètres (denoise, CFG, steps)
5. Cliquez sur "Execute Workflow"
6. Sauvegardez le résultat

---

### Mode 2 : Magic Wand Application

```bash
# Lancer Magic Wand avec intégration ComfyUI
python3 greg_wand.py
```

**Fonctionnalités Magic Wand :**
- ✨ Sélection magique (Magic Wand Tool)
- 🎨 Gestion de calques avancée
- 🖌️ Outils de peinture et masquage
- 🤖 Traitement IA via ComfyUI
- 🎬 Export vidéo

---

### Mode 3 : Autodesk Flame PyBox

```bash
# Installer le handler dans Flame
sudo cp flame_comfyui_pybox.py /opt/Autodesk/flame_2024/python/
sudo cp comfyui_config.json /opt/Autodesk/flame_2024/python/
```

**Dans Flame :**
1. Créer un batch
2. Ajouter un nœud PyBox
3. Sélectionner le handler `flame_comfyui_pybox`
4. Choisir le workflow désiré
5. Connecter vos clips

---

## 🎨 Workflows disponibles

| Workflow | Description | Cas d'usage |
|----------|-------------|-------------|
| **Upscale 4K** | Montée en résolution 4K avec IA | Upscaling, amélioration de détails |
| **Denoise** | Suppression du bruit | Footage ISO élevé, restauration |
| **ControlNet VFX** | Composition avec préservation des contours | VFX, remplacement d'arrière-plan |
| **Frame Interpolation** | Génération de frames intermédiaires | Slow motion, conversion framerate |
| **Color Grade** | Amélioration cinématique des couleurs | Color grading, look film |
| **Texture Enhancement** | Amélioration des détails | Sharpening intelligent, détails |

Voir [README_COMFYUI_INTEGRATION.md](README_COMFYUI_INTEGRATION.md) pour les détails complets.

---

## 💻 Exemples d'utilisation

### Exemple 1 : Traiter une image via Python

```python
import asyncio
from comfyui_client import ComfyUIClient, ComfyUIConfig
import cv2

async def process():
    # Configuration
    config = ComfyUIConfig(host="127.0.0.1", port=8188)
    client = ComfyUIClient(config)

    # Connexion
    await client.connect()

    # Charger workflow
    workflow = client.load_workflow("workflows/upscale_4k_professional.json")

    # Charger et traiter image
    image = cv2.imread("input.jpg")
    results = await client.execute_workflow(workflow, images={"input_image": image})

    # Sauvegarder
    if results:
        cv2.imwrite("output.jpg", results[0])

    await client.disconnect()

asyncio.run(process())
```

### Exemple 2 : Intégration dans Magic Wand

```python
from greg_wand import MainWindow
from greg_wand_comfyui_integration import integrate_comfyui_into_magic_wand

# Créer la fenêtre
window = MainWindow()

# Intégrer ComfyUI
comfyui_widget = integrate_comfyui_into_magic_wand(window)

window.show()
```

**Plus d'exemples :** Voir [example_usage.py](example_usage.py)

---

## 🔧 Configuration

### Fichier : `comfyui_config.json`

```json
{
  "comfyui_host": "127.0.0.1",
  "comfyui_port": 8188,
  "workflows_dir": "./workflows",
  "execution_timeout": 300,
  "workflow_presets": {
    "upscale_4k": {
      "workflow": "upscale_4k_professional.json",
      "description": "Professional 4K upscaling",
      "category": "upscaling",
      "parameters": {
        "denoise_strength": 0.35,
        "cfg_scale": 7.5,
        "steps": 20
      }
    }
  }
}
```

**Paramètres clés :**
- `comfyui_host` : Adresse du serveur ComfyUI
- `comfyui_port` : Port du serveur (défaut: 8188)
- `workflows_dir` : Répertoire des workflows JSON
- `execution_timeout` : Timeout d'exécution (secondes)

---

## 📚 Documentation

### Documentation complète

📖 [README_COMFYUI_INTEGRATION.md](README_COMFYUI_INTEGRATION.md) - **Documentation détaillée**
- Architecture du système
- Guide d'installation complet
- Utilisation avancée
- API et développement
- Dépannage

### Ressources externes

- 🔗 [ComfyUI Documentation](https://docs.comfy.org)
- 🔗 [Autodesk Flame PyBox Docs](https://help.autodesk.com/view/FLAME/2024/ENU/?guid=Flame_API_Pybox_Documentation_html)
- 🔗 [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)

---

## 🐛 Dépannage

### ❌ Problème : "Failed to connect to ComfyUI"

**Solution :**
```bash
# Vérifier que ComfyUI est lancé
curl http://localhost:8188/system_stats

# Vérifier le port dans comfyui_config.json
cat comfyui_config.json | grep port
```

### ❌ Problème : "Workflow not found"

**Solution :**
```bash
# Vérifier les fichiers dans workflows/
ls -la workflows/

# Valider la syntaxe JSON
python3 -m json.tool workflows/upscale_4k_professional.json
```

### ❌ Problème : Dépendances manquantes

**Solution :**
```bash
# Réinstaller les dépendances
pip3 install -r requirements.txt --force-reinstall
```

---

## 🎓 Créer vos propres workflows

1. **Dans ComfyUI GUI** : Créez votre workflow
2. **Exportez** : Menu → "Save (API Format)"
3. **Placez** le fichier dans `./workflows/`
4. **Configurez** dans `comfyui_config.json` :

```json
"mon_workflow": {
  "workflow": "mon_workflow.json",
  "description": "Mon workflow personnalisé",
  "category": "custom",
  "parameters": {
    "denoise_strength": 0.4,
    "cfg_scale": 7.0
  }
}
```

---

## 🙏 Crédits & Remerciements

- **ComfyUI** - [@comfyanonymous](https://github.com/comfyanonymous)
- **Autodesk Flame** - [Autodesk](https://www.autodesk.com)
- **PyQt6** - [Riverbank Computing](https://www.riverbankcomputing.com)
- **OpenCV** - [OpenCV Team](https://opencv.org)

---

## 📜 Licence

Ce projet est fourni tel quel pour usage personnel et professionnel.

**Avertissement :** L'utilisation de modèles IA peut nécessiter des licences spécifiques selon votre juridiction et usage.

---

## 🚀 Roadmap

### Version actuelle : 1.0

✅ Intégration ComfyUI complète
✅ PyBox Handler pour Flame
✅ Interface GUI
✅ 6 workflows professionnels
✅ Documentation complète

### Version future : 2.0 (Prévue)

🔲 Support workflows vidéo AnimateDiff
🔲 Intégration NVIDIA Cosmos
🔲 Traitement batch parallèle
🔲 Export direct vers Flame Timeline
🔲 Cloud rendering support
🔲 Workflows collaboratifs

---

## 📞 Support & Contact

Pour questions, bugs ou suggestions :

1. 📖 Consultez la [documentation détaillée](README_COMFYUI_INTEGRATION.md)
2. 🐛 Vérifiez les [exemples d'utilisation](example_usage.py)
3. 📝 Consultez les logs : `comfyui_integration.log`

---

<div align="center">

**Fait avec ❤️ pour la communauté VFX et les artistes**

⭐ **Si ce projet vous aide, donnez-lui une étoile !** ⭐

</div>

---

## 🎬 Captures d'écran

### Interface GUI - Gestionnaire de workflows
![GUI Manager](docs/screenshots/workflow_manager.png)

### Magic Wand - Intégration ComfyUI
![Magic Wand](docs/screenshots/magic_wand_integration.png)

### Autodesk Flame - PyBox Handler
![Flame PyBox](docs/screenshots/flame_pybox.png)

---

**Bon travail avec ComfyUI + Flame ! 🎬✨**
