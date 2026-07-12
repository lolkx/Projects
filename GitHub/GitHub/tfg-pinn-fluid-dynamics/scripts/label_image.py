from PIL import Image, ImageDraw, ImageFont

# Cargar la imagen original
img = Image.open('Escalon(t=0.65,zoom2),sinetiqueta.png')

# Crear un objeto de dibujo
draw = ImageDraw.Draw(img)

# Definir la nueva etiqueta
nueva_etiqueta = "u(x)"

# Seleccionar una fuente (asegúrate de que esté disponible en tu sistema)
# Puedes usar la fuente predeterminada o cargar una con truetype
try:
    font = ImageFont.truetype("arial.ttf", 26)  # Cambiar tamaño según sea necesario
except IOError:
    font = ImageFont.load_default()

# Añadir la nueva etiqueta en la posición deseada
# Ajusta la posición (x, y) según la imagen
draw.text((15, 254), nueva_etiqueta, font=font, fill="black")  # Posición y color del texto

# Guardar la imagen modificada
img.save('Escalon(t=0.65,zoom2).png')

# Mostrar la imagen modificada
img.show()
