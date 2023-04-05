# ==
# Boilerplate circuit compilation for development
# ==

# Compile circuit
circom ../Backbone1.circom --r1cs --c --prime vesta

# Generate the witness, primarily as a smoke test for the circuit
make -C Backbone1_cpp/
# Create out directory if it doesn't exist
if [ ! -d "../out" ]; then
  echo "Directory 'out' does not exist. Creating directory 'out'."
  mkdir "../out"
fi

# Clean up
mv Backbone1_cpp/Backbone1 ../out
mv Backbone1_cpp/Backbone1.dat ../out
mv Backbone1.r1cs ../out
rm -r Backbone1_cpp/ Backbone1.wtns
