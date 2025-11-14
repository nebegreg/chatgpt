# 🎨 Intégration ComfyUI - Autodesk Flame & Magic Wand

## 📖 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Workflows Disponibles](#workflows-disponibles)
- [Intégration Flame PyBox](#intégration-flame-pybox)
- [API et Développement](#api-et-développement)
- [Dépannage](#dépannage)

---

## 🎯 Vue d'ensemble

Cette intégration permet de traiter des images et des clips vidéo Autodesk Flame via ComfyUI workflows, en utilisant l'intelligence artificielle pour améliorer, restaurer et transformer vos médias.

### Cas d'usage principaux

- **Upscaling 4K/8K** : Montée en résolution avec raffinement IA
- **Denoising** : Suppression du bruit et artefacts de compression
- **VFX Composition** : Composition avancée avec ControlNet
- **Interpolation** : Génération de frames intermédiaires pour slow motion
- **Color Grading** : Amélioration cinématique des couleurs
- **Enhancement** : Amélioration des détails et textures

---

## ✨ Fonctionnalités

### 🔌 Communication WebSocket/REST
- Connexion temps réel avec ComfyUI
- Suivi de progression en direct
- Gestion automatique des reconnexions
- Support multi-threading

### 🎬 Intégration Autodesk Flame
- PyBox handler natif
- Extraction/injection de frames
- Support complet des clips
- Préservation des métadonnées

### 🖼️ Interface Graphique
- Bibliothèque de workflows
- Gestionnaire d'exécution
- Paramètres ajustables
- Logs en temps réel

### 🚀 Workflows Professionnels
- 6+ workflows prêts à l'emploi
- Optimisés pour VFX et post-production
- Paramètres personnalisables
- Export/Import JSON

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AUTODESK FLAME                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Flame PyBox Handler                        │  │
│  │  (flame_comfyui_pybox.py)                           │  │
│  │                                                      │  │
│  │  - Extract frames from clips                        │  │
│  │  - Send to ComfyUI via WebSocket                    │  │
│  │  - Receive processed results                        │  │
│  │  - Inject back into timeline                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │      WebSocket / REST API           │
         │      (comfyui_client.py)            │
         │                                     │
         │  - Queue workflows                  │
         │  - Monitor progress                 │
         │  - Retrieve results                 │
         └─────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │         COMFYUI SERVER              │
         │      (localhost:8188)               │
         │                                     │
         │  - Execute AI workflows             │
         │  - Process images                   │
         │  - Return results                   │
         └─────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │      Workflow Library               │
         │      (./workflows/*.json)           │
         │                                     │
         │  - Upscaling                        │
         │  - Denoising                        │
         │  - VFX Composition                  │
         │  - Frame Interpolation              │
         │  - Color Grading                    │
         │  - Texture Enhancement              │
         └─────────────────────────────────────┘
```

---

## 📦 Installation

### Prérequis

- Python 3.8+
- ComfyUI installé et fonctionnel
- Autodesk Flame (pour l'intégration PyBox)
- Dépendances Python

### Installation rapide

```bash
# 1. Cloner ou télécharger le projet
cd /path/to/project

# 2. Installer les dépendances
chmod +x install_dependencies.sh
./install_dependencies.sh

# 3. Vérifier l'installation
python3 -c "import comfyui_client; print('✓ Installation OK')"
```

### Installation manuelle

```bash
# Installer les dépendances Python
pip3 install -r requirements.txt

# Créer les répertoires nécessaires
mkdir -p workflows
mkdir -p /tmp/flame_comfyui
```

---

## ⚙️ Configuration

### Fichier de configuration : `comfyui_config.json`

```json
{
  "comfyui_host": "127.0.0.1",
  "comfyui_port": 8188,
  "workflows_dir": "./workflows",
  "temp_dir": "/tmp/flame_comfyui",
  "default_resolution": [1920, 1080],
  "execution_timeout": 300
}
```

### Variables importantes

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `comfyui_host` | Adresse du serveur ComfyUI | `127.0.0.1` |
| `comfyui_port` | Port du serveur ComfyUI | `8188` |
| `workflows_dir` | Répertoire des workflows JSON | `./workflows` |
| `execution_timeout` | Timeout d'exécution (secondes) | `300` |

---

## 🚀 Utilisation

### Mode 1 : Interface Graphique Standalone

Lancez le gestionnaire de workflows :

```bash
python3 comfyui_workflow_manager.py
```

**Workflow typique :**

1. **Connexion** : Vérifiez que ComfyUI est connecté (🟢)
2. **Sélection** : Choisissez un workflow dans la bibliothèque
3. **Image** : Chargez une image d'entrée
4. **Paramètres** : Ajustez denoise, CFG, steps
5. **Exécution** : Cliquez sur "Execute Workflow"
6. **Résultat** : Sauvegardez l'image traitée

### Mode 2 : Intégration Magic Wand

Intégrez ComfyUI dans l'application Magic Wand :

```python
from greg_wand import MainWindow
from greg_wand_comfyui_integration import integrate_comfyui_into_magic_wand

# Créer la fenêtre principale
window = MainWindow()

# Intégrer ComfyUI
comfyui_widget = integrate_comfyui_into_magic_wand(window)

window.show()
```

### Mode 3 : Autodesk Flame PyBox

#### Installation du handler

```bash
# Copier le handler dans le répertoire Python de Flame
sudo cp flame_comfyui_pybox.py /opt/Autodesk/flame_2024/python/

# Copier la configuration
sudo cp comfyui_config.json /opt/Autodesk/flame_2024/python/
```

#### Utilisation dans Flame

1. Ouvrez Flame et créez un batch
2. Ajoutez un nœud **PyBox**
3. Sélectionnez le handler `flame_comfyui_pybox`
4. Configurez les paramètres :
   - **Workflow** : Sélectionnez le workflow désiré
   - **Process Mode** : Current Frame / Frame Range / All Frames
5. Connectez vos clips
6. Le PyBox traite automatiquement les frames

---

## 🎨 Workflows Disponibles

### 1. Upscale 4K Professional

**Fichier** : `upscale_4k_professional.json`

**Description** : Monte l'image en résolution 4K avec raffinement IA

**Paramètres** :
- Denoise Strength: `0.35`
- CFG Scale: `7.5`
- Steps: `20`

**Cas d'usage** : Upscaling de footage basse résolution, amélioration de détails

---

### 2. Denoise Professional

**Fichier** : `denoise_professional.json`

**Description** : Supprime le bruit et les artefacts de compression

**Paramètres** :
- Denoise Strength: `0.45`
- CFG Scale: `6.5`
- Steps: `25`

**Cas d'usage** : Footage ISO élevé, vidéo compressée, restauration

---

### 3. ControlNet VFX Composition

**Fichier** : `controlnet_vfx_composition.json`

**Description** : Composition avancée avec préservation des contours (Canny edge)

**Paramètres** :
- Control Strength: `0.8`
- CFG Scale: `8.0`
- Steps: `30`

**Cas d'usage** : VFX, remplacement d'arrière-plan, composition complexe

---

### 4. Video Frame Interpolation

**Fichier** : `video_frame_interpolation.json`

**Description** : Génère des frames intermédiaires pour slow motion fluide

**Paramètres** :
- Blend Factor: `0.5`
- Denoise Strength: `0.25`
- Steps: `20`

**Cas d'usage** : Slow motion, conversion de framerate, fluidification

---

### 5. Color Grade Enhancement

**Fichier** : `color_grade_enhancement.json`

**Description** : Amélioration cinématique des couleurs

**Paramètres** :
- Denoise Strength: `0.25`
- CFG Scale: `5.5`
- Steps: `15`

**Cas d'usage** : Color grading rapide, look cinématique, correction couleur

---

### 6. Texture Detail Enhancement

**Fichier** : `texture_detail_enhancement.json`

**Description** : Améliore les détails et textures de l'image

**Paramètres** :
- Denoise Strength: `0.3`
- CFG Scale: `6.0`
- Steps: `20`

**Cas d'usage** : Amélioration de détails, sharpening intelligent, restauration

---

## 🔧 API et Développement

### Exemple : Traiter une image via Python

```python
import asyncio
from comfyui_client import ComfyUIClient, ComfyUIConfig
import cv2

async def process_image():
    # Configuration
    config = ComfyUIConfig(host="127.0.0.1", port=8188)
    client = ComfyUIClient(config)

    # Connexion
    await client.connect()

    # Charger workflow
    workflow = client.load_workflow("workflows/upscale_4k_professional.json")

    # Charger image
    image = cv2.imread("input.jpg")

    # Exécuter
    results = await client.execute_workflow(
        workflow,
        images={"input_image": image}
    )

    # Sauvegarder
    if results:
        cv2.imwrite("output.jpg", results[0])

    await client.disconnect()

# Run
asyncio.run(process_image())
```

### Exemple : Créer un workflow personnalisé

```python
from comfyui_client import ComfyUIClient

client = ComfyUIClient()

# Charger un workflow de base
workflow = client.load_workflow("workflows/upscale_4k_professional.json")

# Modifier les paramètres
custom_workflow = client.update_workflow_params(workflow, {
    "8": {  # Node ID du KSampler
        "denoise": 0.5,
        "cfg": 8.0,
        "steps": 30
    }
})

# Exécuter le workflow modifié
results = client.execute_workflow(custom_workflow, images={"input_image": my_image})
```

---

## 🐛 Dépannage

### Problème : "Failed to connect to ComfyUI"

**Solution** :
1. Vérifiez que ComfyUI est lancé : `http://localhost:8188`
2. Vérifiez le port dans `comfyui_config.json`
3. Testez manuellement : `curl http://localhost:8188/system_stats`

### Problème : "Workflow not found"

**Solution** :
1. Vérifiez que les fichiers JSON sont dans `./workflows/`
2. Validez la syntaxe JSON : `python3 -m json.tool workflow.json`
3. Rechargez les workflows dans le GUI

### Problème : "Execution timeout"

**Solution** :
1. Augmentez `execution_timeout` dans la config
2. Réduisez le nombre de steps
3. Vérifiez la charge GPU de ComfyUI

### Problème : "PyBox handler not found in Flame"

**Solution** :
1. Copiez `flame_comfyui_pybox.py` dans `/opt/Autodesk/flame_XXXX/python/`
2. Redémarrez Flame
3. Vérifiez les permissions : `chmod 644 flame_comfyui_pybox.py`

---

## 📚 Ressources

### Documentation ComfyUI
- API Reference : https://docs.comfy.org
- Workflows : https://comfyui.org/workflows

### Autodesk Flame
- PyBox Documentation : https://help.autodesk.com/view/FLAME/2024/ENU/?guid=Flame_API_Pybox_Documentation_html
- Python API : https://help.autodesk.com/view/FLAME/2024/ENU/?guid=Flame_API_Python_API_html

---

## 🎉 Workflows Avancés

Pour créer vos propres workflows :

1. Créez votre workflow dans ComfyUI GUI
2. Exportez en format API : **"Save (API Format)"**
3. Placez le fichier JSON dans `./workflows/`
4. Ajoutez une entrée dans `comfyui_config.json` → `workflow_presets`

**Exemple d'entrée preset** :

```json
"mon_workflow": {
  "workflow": "mon_workflow.json",
  "description": "Description de mon workflow",
  "category": "custom",
  "parameters": {
    "denoise_strength": 0.4,
    "cfg_scale": 7.0,
    "steps": 25
  }
}
```

---

## 🙏 Crédits

- **ComfyUI** : https://github.com/comfyanonymous/ComfyUI
- **Autodesk Flame** : https://www.autodesk.com/products/flame
- **PyQt6** : https://www.riverbankcomputing.com/software/pyqt/
- **OpenCV** : https://opencv.org/

---

## 📄 Licence

Ce projet est fourni tel quel pour usage personnel et professionnel.

---

## 🚀 Support

Pour des questions ou problèmes :
1. Vérifiez cette documentation
2. Consultez les logs : `comfyui_integration.log`
3. Testez avec les workflows d'exemple fournis

**Bon travail avec ComfyUI + Flame ! 🎬✨**
