inline void SaveMesh(string str, S3DLMesh & mesh, string tex_name)
{
	//int real = 1;
	unsigned int vsz = (int)mesh.Vertices.size();

	std::fstream f3D(str, std::ios::out);
	f3D << "xof 0302txt 0032" << '\n';
	f3D << "Header {" << '\n';
	f3D << "1;" << '\n';
	f3D << "0;" << '\n';
	f3D << "1;" << '\n';
	f3D << "}" << '\n';
	f3D << "Mesh Scene {" << '\n';
	f3D << vsz << ';' << '\n';

	for (unsigned int i = 0; i < vsz; i++) {
		f3D << mesh.Vertices[i].x << ';';
		f3D << mesh.Vertices[i].y << ';';
		f3D << mesh.Vertices[i].z << ';';
		f3D << ',' << '\n';
	}
	f3D << '\n';
	f3D << mesh.Triangles.size() << ';' << '\n';
	for (unsigned int i = 0; i < unsigned int(mesh.Triangles.size()); i++) {
		f3D << "3;" << mesh.Triangles[i].i[0] << "," << mesh.Triangles[i].i[1] << "," << mesh.Triangles[i].i[2] << ";" << "," << '\n';
	}
	f3D << '\n';

	//f3D << "MeshNormals {" << '\n';
	//f3D << mesh.points.size() << ';'<< '\n';
	//for (unsigned int i = 0; i < mesh.normals.size(); i++){
	//	f3D <<mesh.normals[i].x << ";" << mesh.normals[i].y << ";" << mesh.normals[i].z << ";" << "," <<'\n';
	//	f3D <<mesh.normals[i].x << ";" << mesh.normals[i].y << ";" << mesh.normals[i].z << ";" << "," <<'\n';
	//	f3D <<mesh.normals[i].x << ";" << mesh.normals[i].y << ";" << mesh.normals[i].z << ";" << "," <<'\n';
	//}

	//f3D	<< '\n';
	//f3D << mesh.triangles.size()<< ';'<< '\n';
	//for (unsigned int i = 0; i < mesh.triangles.size(); i++){
	//	f3D << "3;" << mesh.triangles[i].l[0] << ";" << mesh.triangles[i].l[1] << ";" << mesh.triangles[i].l[2] << ";" << "," <<'\n';
	//}
	//f3D << "}" << '\n';
	//f3D	<< '\n';

	if (mesh.TextureCoordinates.size() == mesh.Vertices.size()) {
		f3D << "TextureFilename {" << '\n';
		f3D << '"' << tex_name << '"' << '\n';
		f3D << "}" << '\n';

		f3D << "MeshTextureCoords {" << '\n';
		f3D << mesh.TextureCoordinates.size() << ';' << '\n';

		for (unsigned int i = 0; i < vsz; i++) {
			f3D << mesh.TextureCoordinates[i].x << ';' << mesh.TextureCoordinates[i].y << ';' << ',' << '\n';

		}

		f3D << "	}" << '\n';
	}
	f3D << "}" << '\n';

	f3D.close();
}