#include "ffi.h"
#include "ops.h"

py::dict registrations() {
    py::dict dict;
    dict["render_fwd"] = encapsulate_function(ops::render_fwd::xla);
    return dict;
}

py::bytes make_descriptor(
    unsigned num_points,
    unsigned batch_size,
    std::pair<unsigned, unsigned> img_shape,
    std::pair<float, float> f,
    std::pair<float, float> c,
    float glob_scale,
    float clip_thresh,
    unsigned block_width
) {
    float4 intrins = {f.first, f.second, c.first, c.second};
    dim3 img_shape_dim3 = {img_shape.second, img_shape.first, 1};

    ops::Descriptor desc = {
        num_points,
        batch_size,
        img_shape_dim3,
        intrins,
        glob_scale,
        clip_thresh,
        block_width,
    };

    return pack_descriptor(desc);
}

PYBIND11_MODULE(_jax_gsplat, m) {
    m.def("registrations", &registrations);

    m.def(
        "make_descriptor",
        make_descriptor,
        py::arg("num_points"),
        py::arg("batch_size"),
        py::arg("img_shape"),
        py::arg("f"),
        py::arg("c"),
        py::arg("glob_scale"),
        py::arg("clip_thresh"),
        py::arg("block_width")
    );
}
