# ==
# Boilerplate circuit compilation for development
# ==

# Compile circuit
circom ../Backbone3.circom --r1cs --c --prime vesta

# Generate the witness, primarily as a smoke test for the circuit

# Create out directory if it doesn't exist
if [ ! -d "../out" ]; then
  echo "Directory 'out' does not exist. Creating directory 'out'."
  mkdir "../out"
fi

# Clean up
mv Backbone3_cpp/Backbone3 ../out
mv Backbone3_cpp/Backbone3.dat ../out
mv Backbone3.r1cs ../out
rm -r Backbone3_cpp/ Backbone3.wtns
