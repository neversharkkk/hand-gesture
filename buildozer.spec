[app]

title = Hand Gesture Recognition
package.name = handgesture
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,task
source.exclude_exts = spec
version = 1.3.0
requirements = python3,kivy
orientation = portrait
fullscreen = 0
android.permissions = CAMERA
android.api = 33
android.minapi = 21
android.sdk = 33
android.ndk = 25b
android.accept_sdk_license = True
android.entrypoint = org.kivy.android.PythonActivity
android.enable_androidx = True
android.allow_backup = True
android.skip_update = False
p4a.branch = master
p4a.bootstrap = sdl2
p4a.archs = arm64-v8a
buildozer_dir = .buildozer
bin_dir = ./bin
log_level = 2
show_warnings = True

[buildozer]

search_path = 
log_2_to_stderr = None
