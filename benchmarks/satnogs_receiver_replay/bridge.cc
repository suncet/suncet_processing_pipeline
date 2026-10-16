#include <pybind11/pybind11.h>
#include <gnuradio/satnogs/ax25_decoder.h>
#include <gnuradio/satnogs/frame_decoder.h>
#include <gnuradio/satnogs/doppler_correction_cc.h>
#include <gnuradio/satnogs/dummy_doppler_correction.h>
namespace py = pybind11;
namespace sn = gr::satnogs;
PYBIND11_MODULE(suncet_receiver_native, m) {
  py::module::import("gnuradio.gr");
  py::class_<sn::decoder, std::shared_ptr<sn::decoder>>(m, "decoder");
  py::class_<sn::ax25_decoder, sn::decoder, std::shared_ptr<sn::ax25_decoder>>(m, "ax25_decoder")
    .def_static("make", &sn::ax25_decoder::make);
  py::class_<sn::frame_decoder, gr::sync_block, gr::block, gr::basic_block,
    std::shared_ptr<sn::frame_decoder>>(m, "frame_decoder")
    .def(py::init(&sn::frame_decoder::make));
  py::class_<sn::doppler_correction, sn::doppler_correction::sptr>(m, "doppler_correction");
  py::class_<sn::dummy_doppler_correction, sn::doppler_correction,
    std::shared_ptr<sn::dummy_doppler_correction>>(m, "dummy_doppler_correction")
    .def_static("make", &sn::dummy_doppler_correction::make);
  py::class_<sn::doppler_correction_cc, gr::sync_block, gr::block, gr::basic_block,
    sn::doppler_correction_cc::sptr>(m, "doppler_correction_cc")
    .def(py::init(&sn::doppler_correction_cc::make));
}
