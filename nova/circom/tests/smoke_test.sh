# ==
# Boilerplate circuit compilation for development
# ==

# Compile circuit
circom ../dense_layer.circom --r1cs --wasm # --prime vesta

# Generate the witness, primarily as a smoke test for the circuit
node dense_layer_js/generate_witness.js dense_layer_js/dense_layer.wasm smoke_test.json dense_layer.wtns

# Clean up
mv dense_layer_js/dense_layer.wasm ../out
mv dense_layer.r1cs ../out
rm -r dense_layer_js/ dense_layer.wtns
