[app]

title = Hand Gesture Recognition
package.name = handgesture
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,task
source.exclude_exts = spec
version = 1.0.0
requirements = python3,kivy,opencv,numpy,pyjnius,android
orientation = portrait
fullscreen = 0
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,READ_MEDIA_IMAGES,READ_MEDIA_VIDEO,READ_MEDIA_AUDIO
android.api = 34
android.minapi = 24
android.ndk = 27b
android.sdk = 34
android.accept_sdk_license = True
android.entrypoint = org.kivy.android.PythonActivity
android.enable_androidx = True
android.gradle_dependencies = 
android.skip_update = False
android.allow_backup = True
android.release_artifact = aab
p4a.branch = master
p4a.bootstrap = sdl2
p4a.archs = arm64-v8a
ios.kivy_ios_url = https://github.com/kivy/kivy-ios
ios.kivy_ios_branch = master
ios.ios_deploy_url = https://github.com/phonegap/ios-deploy
ios.ios_deploy_branch = 1.12.2
ios.p4a_branch = master
ios.ios_frameworks = 
buildozer_dir = .buildozer
bin_dir = ./bin
log_level = 2
show_warnings = True

[buildozer]

search_path = 
log_2_to_stderr = None
