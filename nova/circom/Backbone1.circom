pragma circom 2.1.1;
include "./Backbone.circom";

component main { public [step_in] } = Backbone(28, 28, 2, 2, 3, 1, 1);

