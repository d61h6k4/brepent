# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>. All Rights Reserved.
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Serve the model."""
from typing import Dict, List, Tuple

import collections

import jax
import jax.numpy as jnp

import numpy as np

import jraph
import pathlib
import scenepic
import tqdm

import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

from clu import checkpoint
from clu import platform
from ml_collections import config_flags

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties

from brepnet.train import create_model

from brep2graph import graph_from_brep
from brep2graph.utils import load_body
from brep2graph.brepnet_features import scale_solid_to_unit_box

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", None, "Directory stores steps and/or seg files.")
flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True)

COLOURS = [
    "#EA1821", "#F1D3D3", "#F2CB6C", "#3B755F", "#FF2A93", "#7DE0E6",
    "#6B7C95", "#19AAD1", "#FFCC4C", "#FFFFFF", "#FF89B5",
    "#D1BDFF", "#CDB670", "#F8DA1B", "#9A008A", "#7B76A6", "#26CB4D"
]


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return scenepic.Color(*tuple(
        int(value[i:i + lv // 3], 16) / 255. for i in range(0, lv, lv // 3)))


def load_cad(
    stepfile: pathlib.Path
) -> Tuple[jraph.GraphsTuple, collections.defaultdict]:
    loaded_body = load_body(stepfile)

    body = scale_solid_to_unit_box(loaded_body)
    mesh = collections.defaultdict(lambda: {
        "triangulation": collections.deque(),
        "location": None,
        "type": None,
    })

    face_ix = -1
    # create / update a mesh
    # this is important when a mesh has not been display
    # in this case it has no mesh to iterate through

    inc_mesh = BRepMesh_IncrementalMesh(body, 0.8)
    assert inc_mesh.IsDone()

    for face in TopologyExplorer(body, ignore_orientation=True).faces():
        face_ix += 1
        mesh[face_ix]["type"] = "FACE"
        mesh[face_ix]["color"] = COLOURS[0]

        location = TopLoc_Location()
        facing = BRep_Tool.Triangulation(face, location)

        props = GProp_GProps()
        brepgprop_SurfaceProperties(face, props)
        face_centre_of_mass = props.CentreOfMass()
        mesh[face_ix]["location"] = np.array([[
            face_centre_of_mass.X(),
            face_centre_of_mass.Y(),
            face_centre_of_mass.Z()
        ]]) + np.random.rand(1, 3) / 5.

        if facing is not None:
            tri = facing.Triangles()
            nodes = facing.Nodes()

            for i in range(1, facing.NbTriangles() + 1):
                trian = tri.Value(i)
                index1, index2, index3 = trian.Get()

                tria = nodes.Value(index1)
                trib = nodes.Value(index2)
                tric = nodes.Value(index3)
                mesh[face_ix]["triangulation"].append((
                    np.array([tria.X(), tria.Y(), tria.Z()]),
                    np.array([trib.X(), trib.Y(), trib.Z()]),
                    np.array([tric.X(), tric.Y(), tric.Z()]),
                ))

    edge_ix = face_ix
    for edge in TopologyExplorer(body, ignore_orientation=True).edges():
        edge_ix += 1
        mesh[edge_ix]["type"] = "EDGE"
        mesh[edge_ix]["color"] = COLOURS[1]

        props = GProp_GProps()
        brepgprop_LinearProperties(edge, props)
        centre_of_mass = props.CentreOfMass()
        mesh[edge_ix]["location"] = np.array(
            [[centre_of_mass.X(),
              centre_of_mass.Y(),
              centre_of_mass.Z()]]) + np.random.rand(1, 3) / 5.
    coedge_ix = edge_ix
    for wire in TopologyExplorer(body, ignore_orientation=False).wires():
        for coedge in WireExplorer(wire).ordered_edges():
            coedge_ix += 1
            mesh[coedge_ix]["type"] = "COEDGE"
            mesh[coedge_ix]["color"] = COLOURS[2]

            props = GProp_GProps()
            brepgprop_LinearProperties(coedge, props)
            centre_of_mass = props.CentreOfMass()
            mesh[coedge_ix]["location"] = np.array(
                [[centre_of_mass.X(),
                  centre_of_mass.Y(),
                  centre_of_mass.Z()]]) + np.random.rand(1, 3) / 5.

    graph = graph_from_brep(loaded_body)
    jgraph = jraph.GraphsTuple(nodes=graph["nodes"],
                               edges=tf.zeros(graph["n_edge"], tf.float32),
                               receivers=graph["receivers"],
                               senders=graph["senders"],
                               globals=tf.zeros((1, ), tf.uint32),
                               n_node=graph["n_node"],
                               n_edge=graph["n_edge"])

    return (jgraph, mesh)


def load_labels(segfile: pathlib.Path) -> List[int]:
    return list(map(int, segfile.read_text().strip().split('\n')))


def show_cad(scene: scenepic.Scene, canvas: scenepic.Canvas3D,
             storage: collections.defaultdict):
    objects = []
    for ix in storage:
        if storage[ix]["type"] == "FACE":
            mesh = scene.create_mesh()

            for triangle in storage[ix]["triangulation"]:
                mesh.add_triangle(hex_to_rgb(storage[ix]["color"]),
                                  triangle[0],
                                  triangle[1],
                                  triangle[2],
                                  add_wireframe=True)
            mesh.double_sided = True
            objects.append(mesh)
    canvas.create_frame(meshes=objects)


def show_graph(scene: scenepic.Scene, canvas: scenepic.Canvas3D,
               g: jraph.GraphsTuple, storage: collections.defaultdict):

    guard = set()
    objects = []

    for node_from, node_to in zip(g.senders, g.receivers):
        if node_from == node_to:
            continue
        if node_from not in guard:
            node_from_mesh = scene.create_mesh()
            node_from_mesh.add_sphere(
                hex_to_rgb(storage[node_from]["color"]),
                scenepic.Transforms.translate(storage[node_from]["location"])
                @ scenepic.Transforms.scale(1. / 40.))
            objects.append(node_from_mesh)
            guard.add(node_from)

        if node_to not in guard:
            node_to_mesh = scene.create_mesh()
            node_to_mesh.add_sphere(
                hex_to_rgb(storage[node_to]["color"]),
                scenepic.Transforms.translate(storage[node_to]["location"])
                @ scenepic.Transforms.scale(1. / 40.))
            objects.append(node_to_mesh)
            guard.add(node_to)

        if (node_from, node_to) not in guard:
            edge_mesh = scene.create_mesh()
            edge_mesh.add_lines(storage[node_from]["location"],
                                storage[node_to]["location"],
                                hex_to_rgb(storage[node_from]["color"]))
            objects.append(edge_mesh)
            guard.add((node_from, node_to))
            guard.add((node_to, node_from))

    canvas.create_frame(meshes=objects)


def show_info(
    scene: scenepic.Scene,
    canvas: scenepic.Canvas3D,
    labels_name2color: Dict[str, str],
    preds: List[str],
    gts: List[str],
):
    info_frame = canvas.create_frame()
    x = 40
    y = 60

    info_frame.add_text("The colors of the faces:", x - 20, y)
    y += 20
    for label, color in labels_name2color.items():
        info_frame.add_text(f"{ label }", x, y, color=hex_to_rgb(color))
        y += 20

    info_frame.add_text("The prediction and ground truth:", x - 20, y)
    y += 20
    for pred, gt in zip(preds, gts):
        info_frame.add_text(
            f"{pred} {'=' if pred == gt else '!'}= {gt}",
            x,
            y,
            color=hex_to_rgb(COLOURS[-1] if pred == gt else COLOURS[-2]))
        y += 20


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments")

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    # This example only supports single-host training on a single device.
    logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}")
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                         FLAGS.workdir, "workdir")

    checkpoint_dir = pathlib.Path(FLAGS.workdir) / "checkpoints"
    state_dict = checkpoint.load_state_dict(str(checkpoint_dir))
    model = create_model(FLAGS.config, deterministic=True)

    scene = scenepic.Scene()
    cad_canvas = scene.create_canvas_3d(width=590, height=800)
    graph_canvas = scene.create_canvas_3d(width=590, height=800)
    info_canvas = scene.create_canvas_2d(width=200, height=800)
    scene.link_canvas_events(cad_canvas, graph_canvas, info_canvas)

    labels_name = [
        "ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd", "Fillet", "Chamfer",
        "RevolveSide", "RevolveEnd"
    ]
    labels_name2color = {k: v for k, v in zip(labels_name, COLOURS[3:])}

    dataset_dir = pathlib.Path(FLAGS.dataset)
    for stepfile in tqdm.tqdm((dataset_dir / "step").iterdir()):
        if np.random.random_sample() < 0.999:
            continue

        segfile = (dataset_dir / "seg" / stepfile.name).with_suffix(".seg")
        assert segfile.exists()

        labels = []
        if segfile.exists() is not None:
            labels = load_labels(segfile)

        (graph, storage) = load_cad(stepfile)
        preds = [
            labels_name[l]
            for l in jnp.argmax(model.apply(state_dict["params"], graph).nodes,
                                axis=-1)
        ]
        gts = [labels_name[l] for l in labels]

        if gts:
            for ix, gt in enumerate(gts):
                if storage[ix]["type"] == "FACE":
                    storage[ix]["color"] = labels_name2color[gt]
        else:
            for ix, pred in enumerate(preds):
                if storage[ix]["type"] == "FACE":
                    storage[ix]["color"] = labels_name2color[pred]


        show_cad(scene, cad_canvas, storage)
        show_graph(scene, graph_canvas, graph, storage)
        show_info(scene, info_canvas, labels_name2color, preds, gts)

    scene.framerate = 2
    scene.save_as_html("demo.html", title="Show the graph of a CAD")


if __name__ == "__main__":
    flags.mark_flags_as_required(["dataset", "workdir", "config"])
    app.run(main)
