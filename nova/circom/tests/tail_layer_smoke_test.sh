# ==
# Boilerplate circuit compilation for development
# ==

# Compile circuit
circom ../TailLayer.circom --r1cs --wasm # --prime vesta

# Generate the witness, primarily as a smoke test for the circuit
node TailLayer_js/generate_witness.js TailLayer_js/TailLayer.wasm tail_layer_smoke_test.json TailLayer.wtns

# Clean up
mv TailLayer_js/TailLayer.wasm ../out
mv TailLayer.r1cs ../out
rm -r TailLayer_js/ TailLayer.wtns
