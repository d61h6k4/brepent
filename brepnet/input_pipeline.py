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
"""Exposes the Fusion 360 Gallery dataset in a convinent format."""

from __future__ import annotations
from typing import NamedTuple

import functools
import jraph

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

import tfds_fusion_360_gallery_dataset.cad.fusion360gallery.fusion360gallery  # pylint: disable=unused-import


class GraphsTupleSize(NamedTuple):
    """Helper class to represent padding and graph sizes."""
    n_node: int
    n_edge: int
    n_graph: int


def get_raw_datasets() -> dict[str, tf.data.Dataset]:
    """Returns datasets as tf.data.Dataset, organized by split."""
    ds_builder = tfds.builder("fusion360_gallery_segmentation")
    ds_builder.download_and_prepare()
    ds_splits = ["train", "test"]
    datasets = {
        split: ds_builder.as_dataset(split=split)
        for split in ds_splits
    }
    return datasets


def get_datasets(batch_size: int) -> dict[str, tf.data.Dataset]:
    """Returns datasets of batched GraphTuples, organized by split."""
    if batch_size <= 1:
        raise ValueError(
            "Batch size must be > 1 to account for padding graph.")

    datasets = get_raw_datasets()

    # Process each split seperately
    for split_name in datasets:
        datasets[split_name] = datasets[split_name].map(
            convert_to_graph_tuples,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True)

    # Compute the padding budget for the requested batch size.
    budget = estimate_padding_budget_for_batch_size(datasets["train"],
                                                    batch_size,
                                                    num_estimation_graphs=100)

    # Pad an example graph to see what the output shapes will be.
    # We will use this shape information when creating the tf.data.Dataset.
    example_graph = next(datasets["train"].as_numpy_iterator())
    example_padded_graph = jraph.pad_with_graphs(example_graph, *budget)
    padded_graphs_spec = specs_from_graphs_tuple(example_padded_graph)

    # Process each split separately.
    for split_name, dataset_split in datasets.items():

        # Repeat and shuffle the training split.
        if split_name == "train":
            dataset_split = dataset_split.shuffle(
                100, reshuffle_each_iteration=True)
            dataset_split = dataset_split.repeat()

        # Batch and pad each split.
        batching_fn = functools.partial(
            jraph.dynamically_batch,
            graphs_tuple_iterator=iter(dataset_split),
            n_node=budget.n_node,
            n_edge=budget.n_edge,
            n_graph=budget.n_graph)
        dataset_split = tf.data.Dataset.from_generator(
            batching_fn, output_signature=padded_graphs_spec)

        # We cache the test set, since it is small.
        if split_name in ["test"]:
            dataset_split = dataset_split.cache()

        datasets[split_name] = dataset_split
    return datasets


def convert_to_graph_tuples(graph: dict[str, tf.Tensor]) -> jraph.GraphsTuple:
    """Converts a dictionary of tf.Tensor to a GraphsTuple."""

    faces_num = tf.shape(graph["face_labels"])
    # we have 8 different segment names for faces (from 0-7)
    # so let"s give no-face nodes label 8 which is going to "not a face" class.
    labels = tf.concat((tf.cast(graph["face_labels"], tf.int32),
                        8 * tf.ones(graph["n_node"] - faces_num, tf.int32)),
                       axis=0)
    nodes = {"features": graph["nodes"], "labels": labels}
    return jraph.GraphsTuple(nodes=nodes,
                             edges=tf.zeros(graph["n_edge"], tf.float32),
                             receivers=graph["receivers"],
                             senders=graph["senders"],
                             globals=tf.zeros((1, ), tf.uint32),
                             n_node=graph["n_node"],
                             n_edge=graph["n_edge"])


def estimate_padding_budget_for_batch_size(
        dataset: tf.data.Dataset, batch_size: int,
        num_estimation_graphs: int) -> GraphsTupleSize:
    """Estimates the padding budget for a dataset of unbatched GraphsTuples.
  Args:
    dataset: A dataset of unbatched GraphsTuples.
    batch_size: The intended batch size. Note that no batching is performed by
      this function.
    num_estimation_graphs: How many graphs to take from the dataset to estimate
      the distribution of number of nodes and edges per graph.
  Returns:
    padding_budget: The padding budget for batching and padding the graphs
    in this dataset to the given batch size.
  """
    def next_multiple_of_64(val: float):
        """Returns the next multiple of 64 after val."""
        return 64 * (1 + int(val // 64))

    if batch_size <= 1:
        raise ValueError(
            "Batch size must be > 1 to account for padding graphs.")

    total_num_nodes = 0
    total_num_edges = 0
    for graph in dataset.take(num_estimation_graphs).as_numpy_iterator():
        graph_size = get_graphs_tuple_size(graph)
        if graph_size.n_graph != 1:
            raise ValueError("Dataset contains batched GraphTuples.")

        total_num_nodes += graph_size.n_node
        total_num_edges += graph_size.n_edge

    num_nodes_per_graph_estimate = total_num_nodes / num_estimation_graphs
    num_edges_per_graph_estimate = total_num_edges / num_estimation_graphs

    padding_budget = GraphsTupleSize(
        n_node=next_multiple_of_64(num_nodes_per_graph_estimate * batch_size),
        n_edge=next_multiple_of_64(num_edges_per_graph_estimate * batch_size),
        n_graph=batch_size)
    return padding_budget


def specs_from_graphs_tuple(graph: jraph.GraphsTuple):
    """Returns a tf.TensorSpec corresponding to this graph."""
    def get_tensor_spec(array: np.ndarray) -> tf.TensorSpec:
        shape = list(array.shape)
        dtype = array.dtype
        return tf.TensorSpec(shape=shape, dtype=dtype)

    specs = {}
    for field in [
            "nodes", "edges", "senders", "receivers", "globals", "n_node",
            "n_edge"
    ]:
        field_sample = getattr(graph, field)

        if isinstance(field_sample, dict):
            specs[field] = {}
            for k, v in field_sample.items():
                specs[field][k] = get_tensor_spec(v)
        else:
            specs[field] = get_tensor_spec(field_sample)
    return jraph.GraphsTuple(**specs)


def get_graphs_tuple_size(graph: jraph.GraphsTuple):
    """Returns the number of nodes, edges and graphs in a GraphsTuple."""
    return GraphsTupleSize(n_node=np.sum(graph.n_node),
                           n_edge=np.sum(graph.n_edge),
                           n_graph=np.shape(graph.n_node)[0])
