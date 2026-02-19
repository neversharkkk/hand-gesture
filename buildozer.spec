[app]

title = Hand Gesture Recognition
package.name = handgesture
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,txt
source.exclude_exts = spec
version = 2.2.0
requirements = python3,kivy,android
orientation = portrait
fullscreen = 0
android.permissions = CAMERA,RECORD_AUDIO,WRITE_EXTERNAL_STORAGE
android.api = 31
android.minapi = 24
android.sdk = 31
android.ndk = 25b
android.accept_sdk_license = True
android.entrypoint = org.kivy.android.PythonActivity
android.enable_androidx = True
android.allow_backup = True
android.allow_cleartext_traffic = True
p4a.branch = develop
p4a.bootstrap = sdl2
p4a.archs = arm64-v8a
buildozer_dir = .buildozer
bin_dir = ./bin
log_level = 2
show_warnings = True

[buildozer]

search_path = 
log_2_to_stderr = None
