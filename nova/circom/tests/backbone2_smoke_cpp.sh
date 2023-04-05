# ==
# Boilerplate circuit compilation for development
# ==

# Compile circuit
circom ../Backbone2.circom --r1cs --c --prime vesta

# Generate the witness, primarily as a smoke test for the circuit
make -C Backbone2_cpp/
# Create out directory if it doesn't exist
if [ ! -d "../out" ]; then
  echo "Directory 'out' does not exist. Creating directory 'out'."
  mkdir "../out"
fi

# Clean up
mv Backbone2_cpp/Backbone2 ../out
mv Backbone2_cpp/Backbone2.dat ../out
mv Backbone2.r1cs ../out
rm -r Backbone2_cpp/ Backbone2.wtns
