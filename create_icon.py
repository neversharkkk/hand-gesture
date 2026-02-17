import math
from PIL import Image, ImageDraw

def create_prism_icon(size):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    
    scale = size / 256.0
    
    prism_x1 = int(70 * scale)
    prism_y1 = int(50 * scale)
    prism_x2 = int(70 * scale)
    prism_y2 = int(206 * scale)
    prism_x3 = int(160 * scale)
    prism_y3 = int(128 * scale)
    
    prism_points = [(prism_x1, prism_y1), (prism_x2, prism_y2), (prism_x3, prism_y3)]
    
    for i in range(int(8 * scale) + 1, 0, -1):
        offset = i * 0.5
        alpha = int(255 - i * 20)
        outline_color = (100 + i * 5, 100 + i * 5, 120 + i * 5, alpha)
        expanded = [
            (prism_x1 - offset, prism_y1 - offset),
            (prism_x2 - offset, prism_y2 + offset),
            (prism_x3 + offset, prism_y3)
        ]
        draw.polygon(expanded, outline=outline_color)
    
    draw.polygon(prism_points, fill=(30, 30, 40, 255), outline=(60, 60, 80, 255))
    
    for i in range(int(15 * scale)):
        x1 = int((10 + i * 1.5) * scale)
        y_center = int(128 * scale)
        y_offset = int((i - 7.5) * 2 * scale)
        y1 = y_center + y_offset
        
        alpha = int(200 + 55 * (1 - abs(i - 7.5) / 7.5))
        white_color = (255, 255, 255, alpha)
        
        draw.line([(x1, y1), (prism_x1, int(128 * scale))], fill=white_color, width=max(1, int(2 * scale)))
    
    rainbow = [
        (255, 50, 50),
        (255, 150, 50),
        (255, 255, 50),
        (50, 255, 50),
        (50, 200, 255),
        (50, 100, 255),
        (150, 50, 255),
        (200, 50, 200),
    ]
    
    for idx, color in enumerate(rainbow):
        angle = -35 + idx * 10
        rad = math.radians(angle)
        
        for thickness in range(int(5 * scale), 0, -1):
            alpha = int(255 - thickness * 30)
            line_color = color + (alpha,)
            
            x2 = prism_x3 + int((90 + thickness * 2) * scale * math.cos(rad))
            y2 = prism_y3 + int((90 + thickness * 2) * scale * math.sin(rad))
            
            draw.line([(prism_x3, prism_y3), (x2, y2)], fill=line_color, width=max(1, int(3 * scale)))
    
    return img

sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
icons = []
for w, h in sizes:
    icon = create_prism_icon(w)
    icons.append(icon)
    print(f'Created {w}x{h} icon')

icons[0].save('icon.ico', format='ICO', sizes=sizes, append_images=icons[1:])
print('icon.ico created successfully!')
