"""
NICTO AI - Color Palette Tool
Generate color schemes and palettes.
"""

import colorsys
import random
import logging
from typing import Dict, List
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class ColorPaletteTool(Tool):
    """
    Color palette generator.

    Supports:
    - Monochromatic, complementary, analogous, triadic, tetradic
    - HEX, RGB, HSL output
    - Named colors
    - Accessible color combinations
    """

    name = "color_palette"
    description = "Generate color palettes: monochromatic, complementary, analogous, triadic, tetradic."
    parameters = [
        ToolParameter(name="base_color", type="string", description="Base color (HEX: #FF0000 or name: red)", required=False),
        ToolParameter(name="scheme", type="string", description="Color scheme type", required=False, default="analogous", enum=[
            "monochromatic", "complementary", "analogous", "triadic", "tetradic",
            "random", "pastel", "vibrant", "dark",
        ]),
        ToolParameter(name="count", type="integer", description="Number of colors (3-8)", required=False, default=5),
        ToolParameter(name="format", type="string", description="Output format", required=False, default="hex", enum=["hex", "rgb", "hsl"]),
    ]
    tags = ["color", "design", "palette", "visual"]
    timeout_seconds = 5.0

    COLORS = {
        "red": "#FF0000", "blue": "#0000FF", "green": "#00FF00",
        "yellow": "#FFFF00", "purple": "#800080", "orange": "#FFA500",
        "pink": "#FFC0CB", "teal": "#008080", "navy": "#000080",
        "coral": "#FF7F50", "cyan": "#00FFFF", "magenta": "#FF00FF",
        "lime": "#00FF00", "maroon": "#800000", "olive": "#808000",
        "silver": "#C0C0C0", "gray": "#808080", "black": "#000000",
        "white": "#FFFFFF", "indigo": "#4B0082", "violet": "#EE82EE",
        "turquoise": "#40E0D0", "salmon": "#FA8072", "tan": "#D2B48C",
        "gold": "#FFD700", "brown": "#A52A2A", "crimson": "#DC143C",
    }

    def _execute(self, base_color: str = None, scheme: str = "analogous",
                 count: int = 5, format: str = "hex") -> ToolResult:
        count = max(3, min(8, count))

        if base_color:
            hex_color = self._resolve_color(base_color)
            if not hex_color:
                return ToolResult(success=False, error=f"Unknown color: {base_color}")
            r, g, b = self._hex_to_rgb(hex_color)
        else:
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

        if scheme == "monochromatic":
            palette = self._monochromatic(h, s, v, count)
        elif scheme == "complementary":
            palette = self._complementary(h, s, v, count)
        elif scheme == "analogous":
            palette = self._analogous(h, s, v, count)
        elif scheme == "triadic":
            palette = self._triadic(h, s, v, count)
        elif scheme == "tetradic":
            palette = self._tetradic(h, s, v, count)
        elif scheme == "pastel":
            palette = self._pastel(count)
        elif scheme == "vibrant":
            palette = self._vibrant(count)
        elif scheme == "dark":
            palette = self._dark(count)
        else:
            palette = self._random(count)

        colors = []
        for pr, pg, pb in palette:
            if format == "hex":
                colors.append(self._rgb_to_hex(pr, pg, pb))
            elif format == "rgb":
                colors.append(f"rgb({pr}, {pg}, {pb})")
            elif format == "hsl":
                ph, ps, pv = colorsys.rgb_to_hsv(pr / 255, pg / 255, pb / 255)
                pl = (0.299 * pr + 0.587 * pg + 0.114 * pb) / 255 * 100
                colors.append(f"hsl({int(ph * 360)}, {int(ps * 100)}%, {int(pl)}%)")

        # Check accessibility
        accessibility = self._check_accessibility(palette)

        return ToolResult(
            success=True,
            output={
                "base_color": self._rgb_to_hex(r, g, b),
                "scheme": scheme,
                "colors": colors,
                "accessibility": accessibility,
            },
        )

    def _resolve_color(self, color: str) -> str:
        color_lower = color.lower().strip()
        if color_lower.startswith('#'):
            return color_lower[:7]
        return self.COLORS.get(color_lower)

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(self, r: int, g: int, b: int) -> str:
        return f"#{r:02x}{g:02x}{b:02x}"

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> tuple:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _monochromatic(self, h: float, s: float, v: float, count: int) -> List[tuple]:
        return [self._hsv_to_rgb(h, max(0, min(1, s * (0.3 + 0.7 * i / count))), 
                                max(0, min(1, v * (0.3 + 0.7 * i / count)))) 
                for i in range(count)]

    def _complementary(self, h: float, s: float, v: float, count: int) -> List[tuple]:
        colors = [self._hsv_to_rgb(h, s, v)]
        comp_h = (h + 0.5) % 1.0
        for i in range(count - 1):
            ratio = i / (count - 1)
            cur_h = h + (comp_h - h) * ratio
            cur_s = max(0.3, min(1, s * (0.5 + 0.5 * ratio)))
            cur_v = max(0.3, min(1, v * (1.0 - 0.4 * ratio)))
            colors.append(self._hsv_to_rgb(cur_h % 1.0, cur_s, cur_v))
        return colors

    def _analogous(self, h: float, s: float, v: float, count: int) -> List[tuple]:
        spread = 0.1
        return [self._hsv_to_rgb((h - spread + 2 * spread * i / (count - 1)) % 1.0, s, v) 
                for i in range(count)]

    def _triadic(self, h: float, s: float, v: float, count: int) -> List[tuple]:
        angles = [h, (h + 1/3) % 1.0, (h + 2/3) % 1.0]
        return [self._hsv_to_rgb(angles[i % 3], s, v * (0.7 + 0.3 * (i // 3) / max(1, count // 3))) 
                for i in range(count)]

    def _tetradic(self, h: float, s: float, v: float, count: int) -> List[tuple]:
        angles = [h, (h + 0.25) % 1.0, (h + 0.5) % 1.0, (h + 0.75) % 1.0]
        return [self._hsv_to_rgb(angles[i % 4], s, v * (0.7 + 0.3 * (i // 4) / max(1, count // 4))) 
                for i in range(count)]

    def _pastel(self, count: int) -> List[tuple]:
        return [self._hsv_to_rgb(random.random(), random.uniform(0.2, 0.4), random.uniform(0.8, 1.0)) 
                for _ in range(count)]

    def _vibrant(self, count: int) -> List[tuple]:
        return [self._hsv_to_rgb(random.random(), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0)) 
                for _ in range(count)]

    def _dark(self, count: int) -> List[tuple]:
        return [self._hsv_to_rgb(random.random(), random.uniform(0.5, 0.9), random.uniform(0.1, 0.3)) 
                for _ in range(count)]

    def _random(self, count: int) -> List[tuple]:
        return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                for _ in range(count)]

    def _check_accessibility(self, palette: List[tuple]) -> Dict:
        results = {}
        for i, c1 in enumerate(palette):
            for j, c2 in enumerate(palette[i+1:], i+1):
                contrast = self._contrast_ratio(c1, c2)
                key = f"{self._rgb_to_hex(*c1)} vs {self._rgb_to_hex(*c2)}"
                results[key] = {
                    "contrast_ratio": round(contrast, 2),
                    "passes_AA_normal": contrast >= 4.5,
                    "passes_AA_large": contrast >= 3.0,
                    "passes_AAA_normal": contrast >= 7.0,
                }
        return results

    def _contrast_ratio(self, c1: tuple, c2: tuple) -> float:
        def relative_luminance(r, g, b):
            rs = r / 255
            gs = g / 255
            bs = b / 255
            rl = rs / 12.92 if rs <= 0.03928 else ((rs + 0.055) / 1.055) ** 2.4
            gl = gs / 12.92 if gs <= 0.03928 else ((gs + 0.055) / 1.055) ** 2.4
            bl = bs / 12.92 if bs <= 0.03928 else ((bs + 0.055) / 1.055) ** 2.4
            return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl

        l1 = relative_luminance(*c1)
        l2 = relative_luminance(*c2)
        if l1 < l2:
            l1, l2 = l2, l1
        return (l1 + 0.05) / (l2 + 0.05)
