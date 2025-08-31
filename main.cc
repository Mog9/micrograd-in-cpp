#include <iostream>
#include <vector>
#include <unordered_set>
#include <functional>
#include <cmath>
#include <string>

class Value {
public:
    double data;
    double grad;
    std::function<void()> _backward;
    std::unordered_set<Value*> _prev;
    std::string _op;

    Value(double d,
        const std::vector<Value*>& children = {},
        const std::string& op = "")
        : data(d), grad(0.0), _backward([](){}), _prev(children.begin(), children.end()), _op(op) {}


    //add
    static Value* add(Value* a, Value* b) {
        Value* out = new Value(a->data + b->data, {a,b}, "+");

        //backword
        out->_backward = [a,b,out]() {
            a->grad += 1.0 * out->grad;
            b->grad += 1.0 * out->grad;
        };
        return out;
    }


    //mul
    static Value* mul(Value* a, Value* b) {
        Value* out = new Value(a->data * b->data, {a,b}, "*");

        out->_backward = [a,b,out]() {
            a->grad += b->data * out->grad;
            b->grad += a->data * out->grad;
        };
        return out;
    }


    //pow
    static Value* pow(Value* a, double p) {
        Value* out = new Value(std::pow(a->data, p), {a}, "**" + std::to_string(p));

        out->_backward = [a,out,p]() {
            a->grad += p * std::pow(a->data, p - 1) * out->grad;
        };
        return out;
    }

    //relu
    static Value* relu(Value* a) {
        double out_data = (a->data < 0 ? 0.0 : a->data);
        Value* out = new Value(out_data, {a}, "ReLU");

        out->_backward = [a,out,out_data]() {
            a->grad += (out_data > 0 ? 1.0 : 0.0) * out->grad;
        };
        return out;
    }

    void backward() {
        //topological odder
        std::vector<Value*> topo;
        std::unordered_set<Value*> visited;

        std::function<void(Value*)> build_topo = [&](Value* v) {
            if (!visited.count(v)) {
                visited.insert(v);
                for (auto child : v->_prev) build_topo(child);
                topo.push_back(v);
            }
        };
        build_topo(this);

        this->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

    void disp() const {
        std::cout << "Value = " << data << ", grad = " << grad << std::endl;
        if (!_op.empty()) {
            std::cout << "op= " << _op << std::endl;
        }
    }
};

int main() {
    Value* a = new Value(2.0);
    Value* b = new Value(-3.0);
    Value* c = new Value(10.0);


    Value* d = Value::add(Value::mul(a,b), c);
    d->disp();

    d->backward();

    a->disp();
    b->disp();
    c->disp();
    d->disp();
    //actual grad = 0,-3,2,1,1


    delete a; delete b; delete c; delete d;
    return 0;
}
