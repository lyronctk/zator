pragma circom 2.1.1;

template Main() {
    signal input a;
    signal input b; 

    signal input c; 

    a + b === c;
}

component main { public [ a, b ] } = Main();
