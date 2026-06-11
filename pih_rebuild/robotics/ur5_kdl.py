# MODEL: Universal Robots UR5e with PyKDL+pykdl_utils
# AUTHOR: Yi Liu @AiRO 
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

import numpy as np
import PyKDL as kdl

from urdf_parser_py.urdf import URDF
# from kdl_parser.urdf_parser_py.urdf import URDF          # original (ROS1)
# from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model  # original (ROS1)


def _toKdlVector(vec):
    """Convert a list/tuple of 3 floats to a KDL Vector."""
    return kdl.Vector(vec[0], vec[1], vec[2])


def _toKdlRotation(rpy):
    """Convert RPY angles to a KDL Rotation."""
    return kdl.Rotation.RPY(rpy[0], rpy[1], rpy[2])


def _toKdlFrame(urdf_origin):
    """Convert a URDF origin (xyz + rpy) to a KDL Frame."""
    if urdf_origin is None:
        return kdl.Frame.Identity()
    xyz = urdf_origin.xyz if urdf_origin.xyz else [0, 0, 0]
    rpy = urdf_origin.rpy if urdf_origin.rpy else [0, 0, 0]
    return kdl.Frame(_toKdlRotation(rpy), _toKdlVector(xyz))


def _toKdlJoint(urdf_joint):
    """Convert a URDF joint to a KDL Joint."""
    origin = _toKdlFrame(urdf_joint.origin)
    axis = kdl.Vector(0, 0, 0)
    if urdf_joint.axis is not None:
        axis = kdl.Vector(urdf_joint.axis[0], urdf_joint.axis[1], urdf_joint.axis[2])

    if urdf_joint.type == 'revolute' or urdf_joint.type == 'continuous':
        return kdl.Joint(urdf_joint.name, origin.p, origin.M * axis, kdl.Joint.RotAxis)
    elif urdf_joint.type == 'prismatic':
        return kdl.Joint(urdf_joint.name, origin.p, origin.M * axis, kdl.Joint.TransAxis)
    elif urdf_joint.type == 'fixed':
        return kdl.Joint(urdf_joint.name, kdl.Joint.Fixed)
    else:
        return kdl.Joint(urdf_joint.name, kdl.Joint.Fixed)


def _toKdlInertia(urdf_inertia):
    """Convert URDF inertia to KDL RigidBodyInertia (defaults to zero)."""
    if urdf_inertia is None:
        return kdl.RigidBodyInertia()
    origin = _toKdlFrame(urdf_inertia.origin)
    mass = urdf_inertia.mass if urdf_inertia.mass else 0.0
    inertia = urdf_inertia.inertia
    if inertia is None:
        return kdl.RigidBodyInertia(mass, origin.p)
    return kdl.RigidBodyInertia(
        mass, origin.p,
        kdl.RotationalInertia(inertia.ixx, inertia.iyy, inertia.izz,
                              inertia.ixy, inertia.ixz, inertia.iyz)
    )


def kdl_tree_from_urdf_model(robot):
    """
    Build a KDL Tree from a urdf_parser_py URDF model.
    Replacement for pykdl_utils.kdl_parser.kdl_tree_from_urdf_model.
    """
    tree = kdl.Tree(robot.get_root())
    for joint in robot.joints:
        # find the child link
        child_link = None
        for link in robot.links:
            if link.name == joint.child:
                child_link = link
                break
        kdl_joint = _toKdlJoint(joint)
        kdl_frame = _toKdlFrame(joint.origin)
        kdl_inertia = _toKdlInertia(child_link.inertial if child_link else None)
        segment = kdl.Segment(joint.child, kdl_joint, kdl_frame, kdl_inertia)
        tree.addSegment(segment, joint.parent)
    return tree

class URx_kdl():
    def __init__(self, DHfile) -> None:
        robot = URDF.from_xml_file(DHfile)
        tree = kdl_tree_from_urdf_model(robot)
        
        self.chain = tree.getChain("base_link", "tool0")
        # print("the UR5e .urdf model has %d bodies." % tree.getNrOfSegments())
        # print("the UR5e has %d bodies we used to controlled" % chain.getNrOfSegments())
        # print("the UR5e has %d joints we controlled" % chain.getNrOfJoints())

    def forward(self, qpos):
        fk = kdl.ChainFkSolverPos_recursive(self.chain)
        pos = kdl.Frame()
        q = kdl.JntArray(self.chain.getNrOfJoints())
        for i in range(self.chain.getNrOfJoints()):
            q[i] = qpos[i]
        fk_flag = fk.JntToCart(q, pos)
        f_pos = np.zeros(3)
        for i in range(3):
            f_pos[i] = pos.p[i]
        return f_pos
    
    def inverse(self, init_joint, goal_pose, goal_rot):
        try:
            rot = kdl.Rotation()
            rot = rot.Quaternion(goal_rot[0], goal_rot[1], goal_rot[2], goal_rot[3]) # radium x y z w
            pos = kdl.Vector(goal_pose[0], goal_pose[1], goal_pose[2])
        except ValueError:
            print("The target pos can not be transfor to IK-function.")
        target_pos = kdl.Frame(rot, pos)
        # print(target_pos)
        fk = kdl.ChainFkSolverPos_recursive(self.chain)
        #inverse kinematics
        ik_v = kdl.ChainIkSolverVel_pinv(self.chain)
        # ik = kdl.ChainIkSolverPos_NR(chain, fk, ik_v, maxiter=100, eps=math.pow(10, -9))

        # try:
        #     q_min = kdl.JntArray(len(joint_limit_lower))
        #     q_max = kdl.JntArray(len(joint_limit_lower))
        #     for i in range(len(joint_limit_lower)):
        #         q_min[i] = joint_limit_lower[i]
        #         q_max[i] = joint_limit_lower[i]
        # except ValueError:
        #     print("you should input the joint limitation value.")

        # ik_p_kdl = kdl.ChainIkSolverPos_NR_JL(chain, q_min, q_max, fk, ik_v)
        ik_p_kdl = kdl.ChainIkSolverPos_NR(self.chain, fk, ik_v)
        q_init = kdl.JntArray(self.chain.getNrOfJoints())
        for i in range(6):
            q_init[i] = init_joint[i]
        q_out = kdl.JntArray(self.chain.getNrOfJoints())
        ik_p_kdl.CartToJnt(q_init, target_pos, q_out)
        # print("Output angles:", q_out)
        q_out_trans = np.zeros(self.chain.getNrOfJoints())
        for i in range(self.chain.getNrOfJoints()):
            q_out_trans[i] = np.array(q_out[i])
        # print(q_out_trans)
        return (q_out_trans)
