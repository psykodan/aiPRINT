import bpy # type: ignore
import time
import random
import os

dirOut = 'run/'
scene = bpy.context.scene
extruder = scene.objects["Cylinder.001"]
print("randomising scale...")
for e in range(10):
    scale_val = random.randint(1,7)
    extruder.scale = (scale_val,scale_val,0.933)
    extruder.keyframe_insert(data_path="scale",frame=e*100)

print("baking...")
finished = bpy.ops.fluid.bake_all()

if finished:
    print("rendering...")
    os.makedirs(f"{dirOut}/good") 
    os.makedirs(f"{dirOut}/over") 
    os.makedirs(f"{dirOut}/under") 

    name = ""
    scene.frame_set(1)
    for frame in range(scene.frame_start, scene.frame_end + 1):
        res = bpy.data.objects['Cylinder.001'].scale[0]

        if res < 2.5:
            name = "under"
        elif res < 3.5:
            name = "good"
        else:
            name = "over"
            
        scene.render.filepath = dirOut + name + "/" + str(frame).zfill(4)
        scene.frame_set(frame)
        bpy.ops.render.render(write_still=True)

bpy.ops.fluid.free_all()