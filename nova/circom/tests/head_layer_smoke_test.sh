# ==
# Boilerplate circuit compilation for development
# ==

# Compile circuit
circom ../HeadLayer.circom --r1cs --wasm # --prime vesta

# Generate the witness, primarily as a smoke test for the circuit
node HeadLayer_js/generate_witness.js HeadLayer_js/HeadLayer.wasm head_layer_smoke_test.json HeadLayer.wtns

# Clean up
mv HeadLayer_js/HeadLayer.wasm ../out
mv HeadLayer.r1cs ../out
rm -r HeadLayer_js/ HeadLayer.wtns
