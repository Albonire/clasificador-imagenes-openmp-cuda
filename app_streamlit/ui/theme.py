"""Sistema de diseno "moderno neutro": tokens + CSS.

Reemplaza el estilo institucional anterior (degradados rojo/azul, sombras
pesadas, bordes laterales de color, mayusculas con tracking forzado) por una
paleta neutra calmada con un solo acento. Reglas seguidas:

- Sin bordes laterales de color como "acento" (factor-card, alert-warning,
  modelo-status antes usaban `border-left: 4px solid`); en su lugar, fondos
  con tinte sutil o nada.
- Sin combinar borde + sombra ancha ("ghost card"): las tarjetas usan SOLO
  un borde de 1px, sin box-shadow.
- Radios moderados (8-12px), nunca >16px.
- Contraste de texto >= 4.5:1 sobre su fondo (ver COLORS).
- Un unico acento (`accent`); los colores semanticos (despierto/somnoliento)
  se usan solo para el resultado, no decorativamente en toda la interfaz.
"""

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#FAFAFA",  # fondo de pagina (gris neutro, no calido)
    "surface": "#FFFFFF",  # tarjetas, sidebar
    "surface_muted": "#F4F4F5",  # franjas secundarias (pipeline, footer)
    "border": "#E4E4E7",  # bordes 1px
    "ink": "#18181B",  # texto principal (contraste ~17:1 sobre blanco)
    "ink_secondary": "#52525B",  # texto secundario (contraste ~7:1)
    "ink_muted": "#71717A",  # texto terciario / captions (contraste ~4.6:1)
    "accent": "#4F46E5",  # unico acento (indigo-600)
    "accent_soft": "#EEF2FF",  # fondo tenue para el acento
    "success": "#15803D",  # estado "despierto"
    "success_soft": "#F0FDF4",
    "danger": "#B91C1C",  # estado "somnolencia"
    "danger_soft": "#FEF2F2",
    "warning_ink": "#92400E",
    "warning_soft": "#FFFBEB",
}

SPACE = {"xs": "4px", "sm": "8px", "md": "16px", "lg": "24px", "xl": "32px"}
RADIUS = {"sm": "8px", "md": "12px"}

FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700&display=swap"
)


def status_colors(is_drowsy):
    """Color (texto) y fondo tenue para el estado de resultado."""
    if is_drowsy:
        return COLORS["danger"], COLORS["danger_soft"]
    return COLORS["success"], COLORS["success_soft"]


def get_css():
    """Bloque <style> completo, parametrizado por los tokens de COLORS."""
    c = COLORS
    return f"""
<style>
    @import url('{FONT_IMPORT}');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: {c['bg']};
    }}

    /* ---------------------------------------------------------------- */
    /* Cabecera: superficie plana con un divisor de 1px, sin degradado   */
    /* ---------------------------------------------------------------- */
    .header-institucional {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: {RADIUS['md']};
        padding: {SPACE['lg']} {SPACE['xl']};
        margin-bottom: {SPACE['lg']};
    }}
    .header-institucional h1 {{
        margin: 0;
        font-size: 22px;
        font-weight: 600;
        color: {c['ink']};
        letter-spacing: -0.01em;
    }}
    .header-institucional p {{
        margin: {SPACE['xs']} 0 0 0;
        font-size: 14px;
        color: {c['ink_secondary']};
        font-weight: 400;
    }}

    /* ---------------------------------------------------------------- */
    /* Tarjeta de resultado: 1px de borde, SIN sombra (no "ghost card")  */
    /* ---------------------------------------------------------------- */
    .resultado-card {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: {RADIUS['md']};
        padding: {SPACE['lg']};
        margin: {SPACE['md']} 0;
        text-align: center;
    }}
    .resultado-titulo {{
        color: {c['ink_muted']};
        font-size: 12px;
        font-weight: 600;
        margin-bottom: {SPACE['sm']};
        letter-spacing: 0.04em;
    }}
    .resultado-clase {{
        font-size: 28px;
        font-weight: 700;
        margin: {SPACE['sm']} 0;
        line-height: 1.3;
        letter-spacing: -0.01em;
    }}
    .resultado-confianza {{
        color: {c['ink_secondary']};
        font-size: 14px;
        font-weight: 500;
        margin-top: {SPACE['xs']};
    }}

    /* ---------------------------------------------------------------- */
    /* Franja de pasos del pipeline: texto plano + separadores, sin      */
    /* degradado de color                                                */
    /* ---------------------------------------------------------------- */
    .pipeline-flow {{
        background: {c['surface_muted']};
        border: 1px solid {c['border']};
        color: {c['ink_secondary']};
        border-radius: {RADIUS['sm']};
        padding: {SPACE['sm']} {SPACE['md']};
        margin: {SPACE['md']} 0;
        text-align: center;
        font-size: 13px;
        font-weight: 500;
    }}

    .resultado-metricas {{
        border-top: 1px solid {c['border']};
        padding-top: {SPACE['md']};
        margin-top: {SPACE['sm']};
    }}

    .processing-col {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: {RADIUS['sm']};
        padding: {SPACE['sm']};
        text-align: center;
    }}
    .processing-col h4 {{
        color: {c['ink']};
        margin: 0 0 {SPACE['sm']} 0;
        font-size: 13px;
        font-weight: 600;
    }}

    /* ---------------------------------------------------------------- */
    /* Aviso: fondo tenue completo, sin barra lateral de color           */
    /* ---------------------------------------------------------------- */
    .alert-warning {{
        background: {c['warning_soft']};
        border-radius: {RADIUS['sm']};
        padding: {SPACE['sm']} {SPACE['md']};
        margin: {SPACE['sm']} 0;
        color: {c['warning_ink']};
        font-size: 13px;
    }}

    .footer {{
        margin-top: {SPACE['xl']};
        padding-top: {SPACE['md']};
        border-top: 1px solid {c['border']};
        text-align: center;
        color: {c['ink_muted']};
        font-size: 12px;
        line-height: 1.6;
    }}

    .flow-step {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: {RADIUS['sm']};
        padding: {SPACE['sm']} {SPACE['md']};
        margin: {SPACE['xs']} 0;
        font-size: 13px;
        color: {c['ink']};
        text-align: center;
        font-weight: 500;
    }}
    .flow-arrow {{
        text-align: center;
        color: {c['ink_muted']};
        font-size: 14px;
        line-height: 1.2;
    }}

    /* ---------------------------------------------------------------- */
    /* Tarjetas informativas: fondo tenue plano, sin barra lateral       */
    /* ---------------------------------------------------------------- */
    .modelo-status {{
        background: {c['surface_muted']};
        border-radius: {RADIUS['sm']};
        padding: {SPACE['sm']} {SPACE['md']};
        margin: {SPACE['sm']} 0;
        font-size: 13px;
        line-height: 1.8;
        color: {c['ink']};
    }}

    .factor-card {{
        background: {c['surface_muted']};
        border-radius: {RADIUS['sm']};
        padding: {SPACE['sm']} {SPACE['md']};
        margin: {SPACE['xs']} 0;
        font-size: 13px;
        line-height: 1.6;
        color: {c['ink']};
    }}
    .factor-card b {{ color: {c['ink']}; }}

    /* Sidebar clara y legible, sin degradado oscuro */
    section[data-testid="stSidebar"] {{
        background: {c['surface']};
        border-right: 1px solid {c['border']};
    }}
</style>
"""
