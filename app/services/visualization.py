"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import io
from typing import Tuple
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageOps

class VisualizationService:
    """
    Service responsible for rendering chemical structures with high-quality,
    customized visual styles (e.g., Dark Mode / Cyberpunk).
    """

    # --- Rendering Configuration ---
    IMAGE_SIZE: Tuple[int, int] = (1000, 1000)
    BOND_LINE_WIDTH: int = 4
    # FONT_SIZE убрали, доверимся авто-настройке RDKit
    
    # Deep Navy Blue background (RGBA)
    THEME_BACKGROUND_COLOR: Tuple[int, int, int, int] = (15, 17, 26, 255) 

    @staticmethod
    def draw_cyberpunk(mol: Chem.Mol) -> Image.Image:
        """
        Renders a molecule in a 'Cyberpunk' aesthetic.
        """
        # 1. RDKit Rendering (High Resolution)
        d2d = rdMolDraw2D.MolDraw2DCairo(
            VisualizationService.IMAGE_SIZE[0], 
            VisualizationService.IMAGE_SIZE[1]
        )
        
        dopts = d2d.drawOptions()
        
        try:
            dopts.bondLineWidth = VisualizationService.BOND_LINE_WIDTH
            dopts.clearBackground = False
            dopts.additionalAtomLabelPadding = 0.1
        except Exception:
            pass
        
        # Prepare topology and coordinates, then draw
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol)
        d2d.FinishDrawing()
        
        # 2. Image Processing (The "Cyberpunk" Filter)
        png_data = d2d.GetDrawingText()
        foreground_img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        
        # Create solid dark background
        background_img = Image.new("RGBA", foreground_img.size, VisualizationService.THEME_BACKGROUND_COLOR)
        
        # Split channels
        r, g, b, a = foreground_img.split()
        
        # Invert RGB channels (Black atoms/bonds become White/Light)
        rgb_image = Image.merge('RGB', (r, g, b))
        inverted_rgb = ImageOps.invert(rgb_image)
        r_inv, g_inv, b_inv = inverted_rgb.split()
        
        # Recombine: Inverted Colors + Original Alpha
        neon_foreground = Image.merge('RGBA', (r_inv, g_inv, b_inv, a))
        
        # 3. Final Composition
        final_image = Image.alpha_composite(background_img, neon_foreground)
        
        return final_image
