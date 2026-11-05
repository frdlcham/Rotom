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
- Stepped group dimension: a group whose axes jump by a per-axis step, e.g.
  ``[1-3@5x14x14:256:(1,2,2)]`` walks T by 1 and H, W by 2 (5*7*7 = 245
  positions padded to 256). The skipped offsets come from separate dims of the
  same tensor dims, e.g. ``[2:2:1][3:2:1];[1-3@5x14x14:256:(1,2,2)]`` puts each
  (H, W) parity class in its own ciphertext
"""

from math import ceil

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
        steps: For a group dimension, how far each grouped axis jumps per
            step of the group (all 1 for a plain group); else None
    """

    def __init__(
        self, dim, extent, stride=1, dim_type=DimType.FILL, group=None, steps=None
    ):
        """
        Create a dimension.

        Args:
            dim: Tensor dimension index to map to
            extent: Range of indexing (must be a power of 2)
            stride: Step size taken when indexing from [0, extent) (must be power of 2)
            dim_type: Type of dimension (FILL for data, EMPTY for zero-filled)
            group: Logical extents of the grouped tensor dims, if this is a
                group dimension (see ``Dim.group_dim``)
            steps: Per-axis steps of a group dimension (default all 1)

        Raises:
            AssertionError: If extent or stride are not powers of 2, or stride <= 0
        """
        self.dim = dim
        self.extent = extent
        self.stride = stride
        self.dim_type = dim_type
        self.group = tuple(group) if group is not None else None
        self.steps = None
        if self.group is not None:
            steps = (1,) * len(self.group) if steps is None else steps
            self.steps = tuple(int(s) for s in steps)

        if self.dim_type == DimType.EMPTY:
            assert self.dim is None
        if self.group is not None:
            assert isinstance(self.dim, tuple) and len(self.dim) == len(self.group)
            assert len(self.steps) == len(self.group), (self.steps, self.group)
            # a step's skipped offsets are held by a dim of extent ``step``,
            # and dim extents are powers of 2
            assert all(s > 0 and s & (s - 1) == 0 for s in self.steps), self.steps
        else:
            assert steps is None, "steps only apply to group dimensions"
        assert self.stride > 0

    @staticmethod
    def group_dim(start, logical_extents, extent, stride=1, steps=None):
        """Group dims ``start..start+len(logical_extents)-1`` into one dimension.

        Position ``p`` of the group holds the element whose indices in the
        grouped dims are ``np.unravel_index(p, group_counts) * steps`` (see
        ``group_counts``); positions ``p >= prod(group_counts)`` are padding.
        Other dims of the same tensor dims add their index on top, e.g. the
        ``[2:2:1]`` in ``[2:2:1];[1-3@5x14x14:256:(1,2,2)]`` picks the H parity.
        Indices past the logical extents are padding too.
        """
        dims = tuple(range(start, start + len(logical_extents)))
        return Dim(dims, extent, stride, group=logical_extents, steps=steps)

    @property
    def is_group(self):
        """Whether this dimension groups several tensor dims"""
        return self.group is not None

    @property
    def is_stepped(self):
        """Whether this is a group dimension with an axis step > 1"""
        return self.is_group and any(s > 1 for s in self.steps)

    @property
    def group_counts(self):
        """Number of positions the group walks per axis: ``ceil(extent / step)``"""
        return tuple(ceil(e / s) for e, s in zip(self.group, self.steps))

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
        - "[a-b@e_a x .. x e_b:n:(t_a, .., t_b)]" -> stepped group, stride 1
        - "[a-b@e_a x .. x e_b:n:s:(t_a, .., t_b)]" -> stepped group, stride s
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
            # Group dimension: [a-b@e_a x .. x e_b:n:s] or [..:n:(t_a, .., t_b)]
            if "@" in terms[0]:
                if terms[2].strip().startswith("("):
                    return Dim._parse_group(terms[0], terms[1], "1", terms[2])
                return Dim._parse_group(terms[0], terms[1], terms[2], None)
            # Regular filled dimension: [i:n:s]
            return Dim(
                int(terms[0]), int(terms[1]), int(terms[2]), dim_type=DimType.FILL
            )

        # Stepped group dimension with explicit stride: [a-b@..:n:s:(t_a, .., t_b)]
        if len(terms) == 4 and "@" in terms[0]:
            return Dim._parse_group(*terms)

        raise NotImplementedError(string_repr)

    @staticmethod
    def _parse_group(head, extent, stride, steps):
        dims, extents = head.split("@")
        start, end = (int(d) for d in dims.split("-"))
        group = [int(e) for e in extents.split("x")]
        assert len(group) == end - start + 1, (head, group)
        if steps is not None:
            steps = [int(t) for t in steps.strip().strip("()").split(",")]
        return Dim.group_dim(start, group, int(extent), int(stride), steps)

    def __repr__(self):
        """String representation of the dimension"""
        match self.dim_type:
            case DimType.FILL:
                if self.is_group:
                    group = "x".join(str(e) for e in self.group)
                    head = f"[{self.dim[0]}-{self.dim[-1]}@{group}:{self.extent}"
                    if not self.is_stepped:
                        return f"{head}:{self.stride}]"
                    steps = "(" + ",".join(str(t) for t in self.steps) + ")"
                    if self.stride == 1:
                        return f"{head}:{steps}]"
                    return f"{head}:{self.stride}:{steps}]"
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
            self.steps,
        )
