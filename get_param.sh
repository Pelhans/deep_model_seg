mkdir param;
python get_model_var.py;
pushd param;
rm *_Adam*;
mv linear_act_1.bias linear_act.bias; 
mv linear_act_1.weight linear_act.weight
popd;
