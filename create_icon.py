import math
from PIL import Image, ImageDraw

def create_prism_icon(size):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    
    scale = size / 256.0
    
    cx, cy = size // 2, size // 2
    
    prism_h = int(180 * scale)
    prism_w = int(80 * scale)
    prism_x = int(60 * scale)
    
    p1 = (prism_x, cy - prism_h // 2)
    p2 = (prism_x, cy + prism_h // 2)
    p3 = (prism_x + prism_w, cy)
    
    for i in range(int(6 * scale), 0, -1):
        alpha = int(180 - i * 25)
        glow_color = (80, 80, 120, alpha)
        offset = i * int(1.5 * scale)
        expanded = [
            (p1[0] - offset, p1[1] - offset // 2),
            (p2[0] - offset, p2[1] + offset // 2),
            (p3[0] + offset, p3[1])
        ]
        draw.polygon(expanded, fill=None, outline=glow_color)
    
    draw.polygon([p1, p2, p3], fill=(25, 25, 35, 255), outline=(70, 70, 90, 255))
    
    inner_offset = int(8 * scale)
    inner_p1 = (p1[0] + inner_offset, p1[1] + inner_offset)
    inner_p2 = (p2[0] + inner_offset, p2[1] - inner_offset)
    inner_p3 = (p3[0] - inner_offset // 2, p3[1])
    draw.polygon([inner_p1, inner_p2, inner_p3], fill=(35, 35, 50, 255))
    
    beam_start_x = int(15 * scale)
    beam_end_x = p1[0]
    beam_width = int(12 * scale)
    
    for i in range(beam_width, 0, -1):
        alpha = int(255 * (1 - i / beam_width) * 0.8)
        y_offset = int(i * 0.8)
        draw.polygon([
            (beam_start_x, cy - y_offset),
            (beam_end_x, cy),
            (beam_start_x, cy + y_offset)
        ], fill=(255, 255, 255, alpha))
    
    draw.polygon([
        (beam_start_x, cy - beam_width // 2),
        (beam_end_x, cy),
        (beam_start_x, cy + beam_width // 2)
    ], fill=(255, 255, 255, 220))
    
    rainbow_colors = [
        (255, 60, 60),
        (255, 165, 50),
        (255, 255, 60),
        (60, 255, 60),
        (60, 200, 255),
        (80, 120, 255),
        (160, 80, 255),
        (220, 80, 200),
    ]
    
    start_angle = -40
    end_angle = 35
    angle_step = (end_angle - start_angle) / len(rainbow_colors)
    ray_length = int(110 * scale)
    ray_width = max(2, int(5 * scale))
    
    for idx, color in enumerate(rainbow_colors):
        angle = start_angle + idx * angle_step
        rad = math.radians(angle)
        
        for thickness in range(ray_width, 0, -1):
            alpha = int(255 - thickness * 40)
            line_color = color + (max(100, alpha),)
            
            length_factor = 1 + thickness * 0.02
            x_end = p3[0] + int(ray_length * length_factor * math.cos(rad))
            y_end = p3[1] + int(ray_length * length_factor * math.sin(rad))
            
            line_w = max(1, int(3 * scale))
            draw.line([p3, (x_end, y_end)], fill=line_color, width=line_w)
    
    return img

sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
icons = []
for w, h in sizes:
    icon = create_prism_icon(w)
    icons.append(icon)
    print(f'Created {w}x{h} icon')

icons[0].save('icon.ico', format='ICO', sizes=sizes, append_images=icons[1:])
print('icon.ico created successfully!')
