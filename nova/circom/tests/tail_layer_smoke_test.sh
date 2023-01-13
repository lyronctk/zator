# Boilerplate circuit compilation for development
# ==

PTAU_FILE=powersOfTau.ptau
ZKEY_FILE=tail_layer.zkey
WITNESS_FILE=tail_layer.wtns
PROOF_FILE=tail_layer_proof.json
PUBLIC_FILE=tail_layer_public.json
VERIFICATION_JSON=tail_layer_verification_key.json

# Compile circuit
circom ../TailLayer.circom --r1cs --wasm # --prime vesta

# Generate the witness, primarily as a smoke test for the circuit
node TailLayer_js/generate_witness.js TailLayer_js/TailLayer.wasm tail_layer_smoke_test.json $WITNESS_FILE

echo "---Starting setup for proof generation---"
# Setup plonk for proof generation
snarkjs plonk setup TailLayer.r1cs $PTAU_FILE $ZKEY_FILE

# echo "---Verifying Zkey---"
# # Verify the Zkey
# snarkjs zkey verify HeadLayer.r1cs $PTAU_FILE $ZKEY_FILE

echo "---Exporting Zkey---"
# Export the Zkey to json
snarkjs zkey export verificationkey $ZKEY_FILE $VERIFICATION_JSON

echo "---Generate the proof---"
# Create the proof
snarkjs plonk prove $ZKEY_FILE $WITNESS_FILE $PROOF_FILE $PUBLIC_FILE

echo "---Verify the proof---"
# Verify the proof
snarkjs plonk verify $VERIFICATION_JSON $PUBLIC_FILE $PROOF_FILE

# Clean up
mv TailLayer_js/TailLayer.wasm ../out
mv TailLayer.r1cs ../out
rm $VERIFICATION_JSON
rm $ZKEY_FILE
rm $PROOF_FILE
rm $PUBLIC_FILE
rm -r TailLayer_js/ $WITNESS_FILE