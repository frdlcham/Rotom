"""
Dimension representation.

Dimensions are used to track tensor dimensions, their extents, strides, and
types. They are fundamental to representing how tensor data is packed into HE vectors.

Key Concepts:

- Dimension extent: The range of values a dimension can take
- Dimension stride: The step size when indexing through a dimension
- Dimension type: Whether a dimension is filled with data or empty (zeroed)
- Group dimension: several consecutive tensor dims laid out row-major in one
  dimension and padded together, e.g. ``[1-3@5x14x14:1024:1]`` holds dims 1..3
  with logical extents 5, 14, 14 (980 elements) padded to 1024 positions
"""

from enum import Enum

from frontends.tensor import *


class DimType(Enum):
    """
    Enumeration defining dimension types for tensor layouts.

    Dimensions can be either filled with actual data or empty (zero-filled).
    This distinction is important for optimization and memory management
    in homomorphic encryption contexts.

    Attributes:
        FILL: Dimension contains actual tensor data
        EMPTY: Dimension is zero-filled (used for padding)
    """

    FILL = 0  # filled dimension
    EMPTY = 1  # zero-filled dimension


class Dim:
    """
    Represents a tensor dimension with extent, stride, and type.

    Attributes:
        dim: The tensor dimension index this Dim represents
        extent: The range of values this dimension can take (must be power of 2)
        stride: The step size when indexing through this dimension (must be power of 2)
        dim_type: Whether this dimension is FILL or EMPTY
        group: For a group dimension, the logical extents of the grouped tensor
            dims (``dim`` is then the tuple of grouped dim indices); else None
    """

    def __init__(self, dim, extent, stride=1, dim_type=DimType.FILL, group=None):
        """
        Create a dimension.

        Args:
            dim: Tensor dimension index to map to
            extent: Range of indexing (must be a power of 2)
            stride: Step size taken when indexing from [0, extent) (must be power of 2)
            dim_type: Type of dimension (FILL for data, EMPTY for zero-filled)
            group: Logical extents of the grouped tensor dims, if this is a
                group dimension (see ``Dim.group_dim``)

        Raises:
            AssertionError: If extent or stride are not powers of 2, or stride <= 0
        """
        self.dim = dim
        self.extent = extent
        self.stride = stride
        self.dim_type = dim_type
        self.group = tuple(group) if group is not None else None

        if self.dim_type == DimType.EMPTY:
            assert self.dim is None
        if self.group is not None:
            assert isinstance(self.dim, tuple) and len(self.dim) == len(self.group)
        assert self.stride > 0

    @staticmethod
    def group_dim(start, logical_extents, extent, stride=1):
        """Group dims ``start..start+len(logical_extents)-1`` into one dimension.

        Position ``p`` of the group holds the element whose indices in the grouped
        dims are ``np.unravel_index(p, logical_extents)``; positions
        ``p >= prod(logical_extents)`` are padding.
        """
        dims = tuple(range(start, start + len(logical_extents)))
        return Dim(dims, extent, stride, group=logical_extents)

    @property
    def is_group(self):
        """Whether this dimension groups several tensor dims"""
        return self.group is not None

    def __hash__(self):
        """Hash the dimension"""
        return hash(str(self))

    def __eq__(self, other):
        """Check if two dimensions are equal"""
        return hash(self) == hash(other)

    def __lt__(self, other):
        """Compare two dimensions"""
        return self.stride < other.stride

    def parse(string_repr):
        """Parser function to transform a string-represented Dimension
        into a Dimension object.

        Supported forms (with brackets included in string_repr):
        - "[i]"           -> Dim(None, i, 1, FILL)
        - "[G:n]"         -> Dim(None, n, 1, EMPTY)
        - "[i:n:s]"       -> Dim(i, n, s, FILL)
        - "[R:n]"         -> Dim(None, n, 1, FILL)      (replication, stride 1)
        - "[R:n:s]"       -> Dim(None, n, s, FILL)      (replication, explicit stride)
        - "[a-b@e_a x .. x e_b:n:s]" -> group of dims a..b (see ``Dim.group_dim``)
        """
        terms = string_repr.replace("[", "").replace("]", "").split(":")

        # Single extent, no dim index
        if len(terms) == 1:
            return Dim(None, int(terms[0]), 1, dim_type=DimType.FILL)

        # Gap / empty dimension
        if len(terms) == 2 and terms[0] == "G":
            return Dim(None, int(terms[1]), 1, dim_type=DimType.EMPTY)

        # Replication dimension with implicit stride 1: [R:n]
        if len(terms) == 2 and terms[0] == "R":
            return Dim(None, int(terms[1]), 1, dim_type=DimType.FILL)

        # Three-term forms
        if len(terms) == 3:
            # Replication dimension with explicit stride: [R:n:s]
            if terms[0] == "R":
                return Dim(None, int(terms[1]), int(terms[2]), dim_type=DimType.FILL)
            # Group dimension: [a-b@e_a x .. x e_b:n:s]
            if "@" in terms[0]:
                dims, extents = terms[0].split("@")
                start, end = (int(d) for d in dims.split("-"))
                group = [int(e) for e in extents.split("x")]
                assert len(group) == end - start + 1, string_repr
                return Dim.group_dim(start, group, int(terms[1]), int(terms[2]))
            # Regular filled dimension: [i:n:s]
            return Dim(
                int(terms[0]), int(terms[1]), int(terms[2]), dim_type=DimType.FILL
            )

        raise NotImplementedError(string_repr)

    def __repr__(self):
        """String representation of the dimension"""
        match self.dim_type:
            case DimType.FILL:
                if self.is_group:
                    group = "x".join(str(e) for e in self.group)
                    return f"[{self.dim[0]}-{self.dim[-1]}@{group}:{self.extent}:{self.stride}]"
                elif self.dim is None:
                    return f"[R:{self.extent}:{self.stride}]"
                else:
                    return f"[{self.dim}:{self.extent}:{self.stride}]"
            case DimType.EMPTY:
                return f"[G:{self.extent}]"

    def copy(self):
        """Copy the dimension"""
        return Dim(
            self.dim,
            self.extent,
            self.stride,
            self.dim_type,
            self.group,
        )
