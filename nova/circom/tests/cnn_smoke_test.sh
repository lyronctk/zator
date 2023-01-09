# ==
# Boilerplate circuit compilation for development
# ==

# Compile circuit
circom ../ConvolutionalLayer.circom --r1cs --wasm # --prime vesta

# Generate the witness, primarily as a smoke test for the circuit
node ConvolutionalLayer_js/generate_witness.js ConvolutionalLayer_js/ConvolutionalLayer.wasm cnn_smoke_test.json ConvolutionalLayer.wtns

# Clean up
mv ConvolutionalLayer_js/ConvolutionalLayer.wasm ../out
mv ConvolutionalLayer.r1cs ../out
rm -r ConvolutionalLayer_js/ ConvolutionalLayer.wtns
