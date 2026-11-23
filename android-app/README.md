# Own GPT Android app

A Capacitor (WebView) wrapper around the Chat and Train pages. The ONNX
model and onnxruntime-web WASM files are bundled in the APK so chat works
offline. The Train page still needs an internet connection and a GitHub
token.

- Application ID: `com.mrkonlinelk.owngpt`
- Minimum Android: 8.0 (API 26)
- The release keystore lives at `/workspace/own-gpt-keystore/` on the
  machine that first signed the app. It is **not** in this repository.

## Build a release APK

You need JDK 21, Node.js, and the Android SDK.

```bash
cd android-app
npm install
npm run assemble          # copies web/, model.onnx, and ORT into www/
npx cap sync android
# If the signing keystore is at /workspace/own-gpt-keystore/, that path
# is picked up automatically. Otherwise:
#   export OWN_GPT_KEYSTORE_PROPERTIES=/path/to/keystore.properties
echo "sdk.dir=$ANDROID_SDK_ROOT" > android/local.properties
npm run build:apk
```

The signed APK is written to
`android/app/build/outputs/apk/release/app-release.apk`.

GitHub Actions can also build it (`Build APK` workflow). A signed release
needs these repository secrets: `OWN_GPT_KEYSTORE_BASE64`,
`OWN_GPT_KEYSTORE_PASSWORD`, `OWN_GPT_KEY_ALIAS`, `OWN_GPT_KEY_PASSWORD`.
Without them, the workflow still uploads an artifact, but it may be
unsigned.
