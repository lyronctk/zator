# Boilerplate circuit compilation for development
# ==

PTAU_FILE=/Users/lyronctk/Documents/projects/zator/nova/circom/tests/powersOfTau28_hez_final_15.ptau
ZKEY_FILE=head_layer.zkey
WITNESS_FILE=head_layer.wtns
PROOF_FILE=head_layer_proof.json
PUBLIC_FILE=head_layer_public.json
VERIFICATION_JSON=head_layer_verification_key.json

# Compile circuit
circom ../HeadLayer.circom --r1cs --wasm --prime vesta

# Generate the witness, primarily as a smoke test for the circuit
node HeadLayer_js/generate_witness.js HeadLayer_js/HeadLayer.wasm head_layer_smoke_test.json $WITNESS_FILE

echo "---Starting setup for proof generation---"
# Setup groth16 for proof generation
yarn run snarkjs groth16 setup ./HeadLayer.r1cs $PTAU_FILE $ZKEY_FILE

# echo "---Verifying Zkey---"
# # Verify the Zkey
# snarkjs zkey verify HeadLayer.r1cs $PTAU_FILE $ZKEY_FILE

echo "---Exporting Zkey---"
# Export the Zkey to json
snarkjs zkey export verificationkey $ZKEY_FILE $VERIFICATION_JSON

echo "---Generate the proof---"
# Create the proof
snarkjs groth16 prove $ZKEY_FILE $WITNESS_FILE $PROOF_FILE $PUBLIC_FILE

# echo "---Verify the proof---"
# # Verify the proof
# snarkjs plonk verify $VERIFICATION_JSON $PUBLIC_FILE $PROOF_FILE

# Clean up
mv HeadLayer_js/HeadLayer.wasm ../out
mv HeadLayer.r1cs ../out
rm $VERIFICATION_JSON
rm $ZKEY_FILE
rm $PROOF_FILE
rm $PUBLIC_FILE
rm -r HeadLayer_js/ $WITNESS_FILE