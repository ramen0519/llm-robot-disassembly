"""Isolated Python 3.10 worker: XCAF occurrence tree, no heuristic solid splitting."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.runtime' / 'ocp'))
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
from OCP.TColStd import TColStd_SequenceOfAsciiString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool as ST


def label_id(label):
    text = TCollection_AsciiString()
    TDF_Tool.Entry_s(label, text)
    return text.ToCString()


def name(label):
    obj = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), obj):
        return obj.Get().ToExtString()
    return None


def mesh(shape, dest):
    BRepMesh_IncrementalMesh(shape, .05, False, .15, False)
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    offset = 0
    with dest.open('x', encoding='ascii') as stream:
        while faces.More():
            face = TopoDS.Face_s(faces.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is None:
                raise ValueError('unmeshed STEP face')
            for i in range(1, tri.NbNodes()+1):
                p = tri.Node(i).Transformed(loc.Transformation())
                stream.write('v %.15g %.15g %.15g\n' % (p.X(), p.Y(), p.Z()))
            for i in range(1, tri.NbTriangles()+1):
                a,b,c = tri.Triangle(i).Get()
                if face.Orientation() == TopAbs_REVERSED:
                    b,c = c,b
                stream.write('f %d %d %d\n' % (a+offset,b+offset,c+offset))
            offset += tri.NbNodes()
            faces.Next()
    if not offset:
        raise ValueError('empty STEP definition')


def convert(source, out):
    doc = TDocStd_Document(TCollection_ExtendedString('BinXCAF'))
    reader = STEPCAFControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError('STEP read failed')
    length, angle, solid = (TColStd_SequenceOfAsciiString() for _ in range(3))
    reader.Reader().FileUnits(length, angle, solid)
    units = [length.Value(i).ToCString() for i in range(1,length.Length()+1)]
    if not units:
        raise ValueError('STEP units unknown')
    reader.ChangeReader().SetSystemLengthUnit(1.0)  # OCCT internal length unit is one mm
    if not reader.Transfer(doc):
        raise ValueError('STEP transfer failed')
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence()
    shapes.GetFreeShapes(roots)
    records, definitions, non_faces = [], {}, []
    def visit(occurrence, parent, trail):
        definition = TDF_Label()
        is_ref = ST.IsReference_s(occurrence)
        if is_ref:
            if not ST.GetReferredShape_s(occurrence, definition):
                raise ValueError('broken STEP occurrence')
            where = parent.Multiplied(ST.GetLocation_s(occurrence))
        else:
            definition, where = occurrence, parent
        key = label_id(definition)
        if key in trail:
            raise ValueError('cyclic XCAF occurrence tree')
        children = TDF_LabelSequence()
        if ST.GetComponents_s(definition, children):
            for idx in range(1,children.Length()+1):
                visit(children.Value(idx), where, trail+[key])
            return
        shape = ST.GetShape_s(definition)
        where = where.Multiplied(shape.Location())
        if not TopExp_Explorer(shape, TopAbs_FACE).More():
            non_faces.append(dict(occurrence=label_id(occurrence),name=name(definition),
                shape_type=str(shape.ShapeType()),reason='no B-rep faces to simulate'))
            return
        if key not in definitions:
            geom = 'shape_%03d' % len(definitions)
            mesh(shape.Located(TopLoc_Location()), out/(geom+'.obj'))
            definitions[key] = geom
        geom = definitions[key]
        tr = where.Transformation()
        mat = [[tr.Value(i,j) for j in range(1,5)] for i in range(1,4)]+[[0,0,0,1]]
        records.append(dict(part_id=str(len(records)), geometry_id=geom, mesh=geom+'.obj',
            cad_name=name(definition), occurrence_name=name(occurrence), definition_label=key,
            occurrence_label=label_id(occurrence), assembly_matrix_mm=mat, mesh_coordinate_frame='part_local'))
    for i in range(1,roots.Length()+1):
        visit(roots.Value(i), TopLoc_Location(), [])
    if not records:
        raise ValueError('no STEP parts')
    with (out/'occurrences.json').open('x') as stream:
        json.dump(dict(unit='mm',step_declared_length_units=units,backend='OCCT XCAF 7.9.3',
                       linear_deflection_mm=.05,angular_deflection_rad=.15,non_surface_occurrences=non_faces,parts=records),stream)
if __name__=='__main__':
    convert(Path(sys.argv[1]),Path(sys.argv[2]))
