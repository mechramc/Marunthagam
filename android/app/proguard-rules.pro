# Marunthagam ProGuard rules
# Applied only to release builds (minifyEnabled = false in debug).

# Keep JNI-called classes so ProGuard doesn't strip the native method declarations
-keep class com.marunthagam.inference.LlamaWrapper { *; }

# Keep data classes used in serialisation / reflection
-keep class com.marunthagam.inference.TriageResult { *; }
-keep class com.marunthagam.inference.TriageLevel { *; }
-keep class com.marunthagam.storage.TriageLog { *; }
