for /l %%x in (1, 1, 100) do (
   echo %%x
   blender --background "sim.blend" --python "headless.py"
   RENAME run %%x
)