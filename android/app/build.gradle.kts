plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.marunthagam"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.marunthagam.marunthagam"
        minSdk = 29          // Android 10 — required for llama.cpp JNI stability + SQLite WAL improvements
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        // arm64-v8a only — production target matches CMakeLists.txt NEON flags;
        // x86_64 omitted to keep APK size down (no emulator support needed for submission)
        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                // Forwards to CMakeLists.txt which builds libmarunthagam.so via llama.cpp
                arguments(
                    "-DANDROID_STL=c++_shared",   // required by llama.cpp C++17 template usage
                    "-DANDROID_PLATFORM=android-29"
                )
                cppFlags("-std=c++17")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    buildFeatures {
        // ViewBinding generates ActivityTriageBinding from activity_triage.xml;
        // no Compose — UI is XML-based to avoid Compose overhead on low-RAM phones
        viewBinding = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        resources {
            // Exclude duplicate META-INF files that conflict across coroutines artifacts
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")
    // CardView — used directly in activity_triage.xml (transitive via appcompat, but explicit
    // here to make the dependency graph readable and survive future appcompat version changes)
    implementation("androidx.cardview:cardview:1.0.0")
    // Coroutines — used by LlamaWrapper and TriageEngine for IO dispatch
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("androidx.core:core-ktx:1.15.0")
}
