# ==
# Boilerplate circuit compilation for development
# ==

LABEL=MiMC3D

# Compile circuit
circom ../${LABEL}.circom --r1cs --wasm --prime vesta

# Generate the witness, primarily as a smoke test for the circuit
node ${LABEL}_js/generate_witness.js ${LABEL}_js/${LABEL}.wasm ${LABEL}_smoke.json ${LABEL}.wtns

# Clean up
mv ${LABEL}_js/${LABEL}.wasm ../out
mv ${LABEL}.r1cs ../out
rm -r ${LABEL}_js/ ConvolutionalLayer.wtns
