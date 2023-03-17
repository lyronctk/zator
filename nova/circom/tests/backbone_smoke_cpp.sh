# ==
# Boilerplate circuit compilation for development
# ==

# Compile circuit
circom ../Backbone.circom --r1cs --c --prime vesta

# Generate the witness, primarily as a smoke test for the circuit
make -C Backbone_cpp/
./Backbone_cpp/Backbone backbone_smoke.json Backbone.wtns

# Create out directory if it doesn't exist
if [ ! -d "../out" ]; then
  echo "Directory 'out' does not exist. Creating directory 'out'."
  mkdir "../out"
fi

# Clean up
mv Backbone_cpp/Backbone ../out
mv Backbone_cpp/Backbone.dat ../out
mv Backbone.r1cs ../out
rm -r Backbone_cpp/ Backbone.wtns
