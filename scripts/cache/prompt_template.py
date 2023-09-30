all_classes = ['stingray', 'hen', 'magpie', 'kite', 'vulture',
               'agama',   'tick', 'quail', 'hummingbird', 'koala',
               'jellyfish', 'snail', 'crawfish', 'flamingo', 'orca',
               'chihuahua', 'coyote', 'tabby', 'leopard', 'lion',
               'tiger','ladybug', 'fly' , 'ant', 'grasshopper',
               'monarch', 'starfish', 'hare', 'hamster', 'beaver',
               'zebra', 'pig', 'ox', 'impala',  'mink',
               'otter', 'gorilla', 'panda', 'sturgeon', 'accordion',
               'carrier', 'ambulance', 'apron', 'backpack', 'balloon',
               'banjo','barn','baseball', 'basketball', 'beacon',
               'binder', 'broom', 'candle', 'castle', 'chain',
               'chest', 'church', 'cinema', 'cradle', 'dam',
               'desk', 'dome', 'drum','envelope', 'forklift',
               'fountain', 'gown', 'hammer','jean', 'jeep',
               'knot', 'laptop', 'mower', 'library','lipstick',
               'mask', 'maze', 'microphone','microwave','missile',
                'nail', 'perfume','pillow','printer','purse',
               'rifle', 'sandal', 'screw','stage','stove',
               'swing','television','tractor','tripod','umbrella',
                'violin','whistle','wreck', 'broccoli', 'strawberry'
               ]

def prompt1(text):
    text = text + ', wildlife photography, photograph, high quality, wildlife, f 1.8, soft focus, 8k, national geographic, award - winning photograph by nick nichols'
    return text

def prompt2(text):
    text = text + ', by James McDonald and Joarc Architects, home, interior, octane render, deviantart, cinematic, key art, hyperrealism, sun light, sunrays, canon eos c 300, ƒ 1.8, 35 mm, 8k, medium - format print'
    return text

def prompt3(text):
    text = text + ', shot 35 mm, realism, octane render, 8k, trending on artstation, 35 mm camera, unreal engine, hyper detailed, photo - realistic maximum detail, volumetric light, realistic matte painting, hyper photorealistic, trending on artstation, ultra - detailed, realistic'
    return text


def prompt4(text):
    text = text + ', anthro, very cute kid\'s film character, disney pixar zootopia character concept artwork, 3d concept, detailed fur, high detail iconic character for upcoming film, trending on artstation, character design, 3d artistic render, highly detailed, octane, blender, cartoon, shadows, lighting'
    return text

def prompt5(text):
    text = text + ', character sheet, concept design, contrast, style by kim jung gi, zabrocki, karlkka, jayison devadas, trending on artstation, 8k, ultra wide angle, pincushion lens effect'
    return text


def prompt6(text):
    text = text + ', cyberpunk, in heavy raining futuristic tokyo rooftop cyberpunk night, sci-fi, fantasy, intricate, very very beautiful, elegant, neon light, highly detailed, digital painting, artstation, concept art, soft light, hdri, smooth, sharp focus, illustration, art by tian zi and craig mullins and wlop and alphonse much'
    return text

def prompt7(text):
    text = text + ', ultra realistic, concept art, intricate details, highly detailed, photorealistic, octane render, 8k, unreal engine, sharp focus, volumetric lighting unreal engine. art by artgerm and alphonse mucha'
    return text


def prompt8(text):
    text = text + ', epic concept art by barlowe wayne, ruan jia, light effect, volumetric light, 3d, ultra clear detailed, octane render, 8k, dark green'
    return text

def prompt9(text):
    text = text + ', cute, funny, centered, award winning watercolor pen illustration, detailed, disney, isometric illustration, drawing, by Stephen Hillenburg, Matt Groening, Albert Uderzo'
    return text

def prompt10(text):
    text = 'photograph of a Fashion model, ' + text + ', full body, highly detailed and intricate, golden ratio, vibrant colors, hyper maximalist, futuristic, city background, luxury, elite, cinematic, fashion, depth of field, colorful, glow, trending on artstation, ultra high detail, ultra realistic, cinematic lighting, focused, 8k'
    return text


def prompt11(text):
    text = text + ', birds in the sky, waterfall close shot 35 mm, realism, octane render, 8 k, exploration, cinematic, trending on artstation, 35 mm camera, unreal engine, hyper detailed, photo - realistic maximum detail, volumetric light, moody cinematic epic concept art, realistic matte painting, hyper photorealistic, epic, trending on artstation, movie concept art, cinematic composition, ultra - detailed, realistic'
    return text

def prompt12(text):
    text = text + ', depth of field. bokeh. soft light. by Yasmin Albatoul, Harry Fayt. centered. extremely detailed. Nikon D850, (35mm|50mm|85mm). award winning photography.'
    return text


def prompt13(text):
    text = 'portrait photo of ' + text + ', photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography'
    return text

def prompt14(text):
    text = text + ', fog, animals, birds, deer, bunny, postapocalyptic, overgrown with plant life and ivy, artgerm, yoshitaka amano, gothic interior, 8k, octane render, unreal engine'
    return text

def prompt15(text):
    text = '23rd century scientific schematics for ' + text + ', blueprint, hyperdetailed vector technical documents, callouts, legend, patent registry'
    return text
def prompt16(text):
    text = text + ', sketch, drawing, detailed, pencil, black and white by Adonna Khare, Paul Cadden, Pierre-Yves Riveau'
    return text

def prompt17(text):
    text = text + ', by Andrew McCarthy, Navaneeth Unnikrishnan, Manuel Dietrich, photo realistic, 8 k, cinematic lighting, hd, atmospheric, hyperdetailed, trending on artstation, deviantart, photography, glow effect'
    return text

def prompt18(text):
    text = 'sprite of video games ' + text + 'icons, 2d icons, rpg skills icons, world of warcraft, league of legends, ability icon, fantasy, potions, spells, objects, flowers, gems, swords, axe, hammer, fire, ice, arcane, shiny object, graphic design, high contrast, artstatio'
    return text

def prompt19(text):
    text = text + ', steampunk cybernetic biomechanical, 3d model, very coherent symmetrical artwork, unreal engine realistic render, 8k, micro detail, intricate, elegant, highly detailed, centered, digital painting, artstation, smooth, sharp focus, illustration, artgerm, Caio Fantini, wlop'
    return text

def prompt20(text):
    text = 'photograph of ' + text + ', photorealistic, vivid, sharp focus, reflection, refraction, sunrays, very detailed, intricate, intense cinematic composition'
    return text

pool = [prompt1, prompt2, prompt3, prompt4, prompt5,
        prompt6, prompt7, prompt8, prompt9, prompt10,
        prompt11, prompt12, prompt13, prompt14, prompt15,
        prompt16, prompt17, prompt18, prompt19, prompt20,]

import random
def prompt_engineering(text):
    func = random.sample(pool, 1)
    # breakpoint()
    return func[0](text)


# prompt_engineering('hhh')