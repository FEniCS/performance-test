
#pragma once

#include <catalyst.hpp>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>

#include <iostream>
#include <string>

namespace CatalystAdaptor
{

void Initialize(std::string script)
{
  conduit_cpp::Node node;
  node["catalyst/scripts/script0"].set_string(script.c_str());
  node["catalyst_load/implementation"] = "paraview";
  node["catalyst_load/search_paths/paraview"] = PARAVIEW_IMPL_DIR;
  catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
  {
    std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
  }
}

void Execute(int cycle, double time, dolfinx::fem::Function<double>& u)
{
  conduit_cpp::Node exec_params;

  // add time/cycle information
  auto state = exec_params["catalyst/state"];
  state["timestep"].set(cycle);
  state["time"].set(time);
  state["multiblock"].set(1);

  // Add channels.
  // We only have 1 channel here. Let's name it 'grid'.
  auto channel = exec_params["catalyst/channels/grid"];

  // Since this example is using Conduit Mesh Blueprint to define the mesh,
  // we set the channel's type to "mesh".
  channel["type"].set("mesh");

  // now create the mesh.
  auto mesh = channel["data"];

  // start with coordsets (of course, the sequence is not important, just make
  // it easier to think in this order).
  mesh["coordsets/coords/type"].set("explicit");

  std::shared_ptr<const dolfinx::mesh::Mesh> input_mesh
      = u.function_space()->mesh();
  const dolfinx::mesh::Geometry& geometry = input_mesh->geometry();
  const int np
      = geometry.index_map()->size_local() + geometry.index_map()->num_ghosts();
  mesh["coordsets/coords/values/x"].set_external(geometry.x().data(), np,
                                                 0,
                                                 3 * sizeof(double));
  mesh["coordsets/coords/values/y"].set_external(geometry.x().data(), np,
                                                 sizeof(double),
                                                 3 * sizeof(double));
  mesh["coordsets/coords/values/z"].set_external(geometry.x().data(), np,
                                                 2 * sizeof(double),
                                                 3 * sizeof(double));

  // Add topology
  const auto& topology = input_mesh->topology();
  int tdim = topology.dim();
  int ncells = topology.index_map(tdim)->size_local()
               + topology.index_map(tdim)->num_ghosts();
  std::span<const std::int32_t> conn(
      topology.connectivity(tdim, 0)->array().data(), 4 * ncells);
  std::vector<int> geom_conn
      = dolfinx::mesh::entities_to_geometry(*input_mesh, 0, conn, false);

  mesh["topologies/mesh/type"].set("unstructured");
  mesh["topologies/mesh/coordset"].set("coords");
  mesh["topologies/mesh/elements/shape"].set("tet");
  mesh["topologies/mesh/elements/connectivity"].set_external(geom_conn.data(),
                                                             geom_conn.size());

  // Add field - this is easy for P1 as it coincidentally has the same DofMap as the geometry
  auto fields = mesh["fields"];
  fields["u/association"].set("vertex");
  fields["u/topology"].set("mesh");
  fields["u/volume_dependent"].set("false");
  fields["u/values"].set_external(u.x()->array().data(), np);

  catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
  if (err != catalyst_status_ok)
  {
    std::cerr << "Failed to execute Catalyst: " << err << std::endl;
  }
}

void Finalize()
{
  conduit_cpp::Node node;
  catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
  {
    std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
  }
}
} // namespace CatalystAdaptor
