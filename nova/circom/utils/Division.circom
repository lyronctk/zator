// credit: https://github.com/thomasthechen/zk-ml-applications/blob/main/circuits/division.circom

pragma circom 2.1.1;

include "../node_modules/circomlib-ml/circuits/util.circom";
include "../node_modules/circomlib-ml/circuits/circomlib/sign.circom";
include "../node_modules/circomlib-ml/circuits/circomlib/comparators.circom";
include "../node_modules/circomlib-ml/circuits/circomlib/bitify.circom";

// NB: RangeProof is inclusive.
// input: field element, whose abs is claimed to be <= than max_abs_value
// output: none
// also checks that both max and abs(in) are expressible in `bits` bits
template RangeProof(bits) {
    signal input in; 
    signal input max_abs_value;

    /* check that both max and abs(in) are expressible in `bits` bits  */
    component n2b1 = Num2Bits(bits+1);
    n2b1.in <== in + (1 << bits);
    component n2b2 = Num2Bits(bits);
    n2b2.in <== max_abs_value;

    /* check that in + max is between 0 and 2*max */
    component lowerBound = LessThan(bits+1);
    component upperBound = LessThan(bits+1);

    lowerBound.in[0] <== max_abs_value + in; 
    lowerBound.in[1] <== 0;
    lowerBound.out === 0;

    upperBound.in[0] <== 2 * max_abs_value;
    upperBound.in[1] <== max_abs_value + in; 
    upperBound.out === 0;
}

template Division(divisor) {
  signal input dividend; 
  signal output quotient;

  component is_neg = IsNegative();
  is_neg.in <== dividend;

  signal is_dividend_negative;
  is_dividend_negative <== is_neg.out;

  signal dividend_adjustment;
  dividend_adjustment <== 1 + is_dividend_negative * -2; // 1 or -1

  signal abs_dividend;
  signal abs_quotient;
  abs_dividend <== dividend * dividend_adjustment; // 8
  abs_quotient <-- abs_dividend \ divisor;

  signal abs_product;
  abs_product <== abs_quotient * divisor;

  var N_BITS = 250;
  component quotientUpper = LessEqThan(N_BITS);
  quotientUpper.in[0] <== abs_product;
  quotientUpper.in[1] <== abs_dividend;
  quotientUpper.out === 1;

  component quotientLower = LessThan(N_BITS);
  quotientLower.in[0] <== abs_dividend - divisor;
  quotientLower.in[1] <== abs_product;
  quotientLower.out === 1;

  component quotientCheck = LessEqThan(N_BITS);
  // log(abs_quotient);
  // log(abs_dividend);
  // log();
  quotientCheck.in[0] <== abs_quotient;
  quotientCheck.in[1] <== abs_dividend;
  quotientCheck.out === 1;

  quotient <== abs_quotient * dividend_adjustment;

  component checkRange = RangeProof(N_BITS);
  checkRange.in <== abs_dividend;
  checkRange.max_abs_value <== 10**30;
}
