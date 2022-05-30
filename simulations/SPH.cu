#include "../src/rendering/Application.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <glm/glm.hpp>

#include "../src/particle_sys/particleSys.h"
#include "../src/particle_sys/sphParticleSys.h"
#include "../src/rendering/particleSysRenderer.h"

constexpr int NUM_PARTICLES = 10000;
#define TWOPI 6.2831853072

struct MyApp : public Application {
	SphParticleSys psys;
	ParticleSystemRenderer psr;


	int sim_frames_per_frame = 1;
	bool run_simulation = true;

	MyApp(int width, int height, std::string title) : Application(width, height, std::move(title)), 
		psys(NUM_PARTICLES, glm::vec3(-50.0, -50.0, -50.0), glm::vec3(50.0, 50.0, 50.0), { }),
		psr(&psys) {
		psr.radius = 21.6;
	}
	void run() {
		Timer dtTimer;
		while (!glfwWindowShouldClose(window)) {

			Timer timer;

			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::SetNextWindowSize(ImVec2(m_width/4, m_height/2)); // ensures ImGui fits the GLFW window
			ImGui::SetNextWindowPos(ImVec2(0, 0));

			ImGui::Begin("Info/Settings", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
			ImGui::Text("Visualization");
			ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::DragInt("Simulation frames per frame", &sim_frames_per_frame, 1, 1, 20);
			ImGui::DragFloat("Zoom", &psr.zoom, 0.01, 0.5, 3.0);
			ImGui::DragFloat("Particle radius", &psr.radius, 0.1, 0.3, 50.0);
			ImGui::DragFloat("X rotation", &psr.xrot, 0.01, -TWOPI, TWOPI);
			ImGui::DragFloat("Y rotation", &psr.yrot, 0.01, -TWOPI, TWOPI);
			ImGui::DragFloat("Z rotation", &psr.zrot, 0.01, -TWOPI, TWOPI);
			ImGui::Checkbox("Show simulation box", &psr.show_cube);
			ImGui::Text("Simulation");
			ImGui::Checkbox("Running simulation", &run_simulation);
			bool sph_changed = false;
			sph_changed |= ImGui::DragFloat("Ext force strength", &psys.h_bss->ExtForce.y, 0.1, -600.0, 600.0);
			sph_changed |= ImGui::DragFloat("Radius of simulation", &psys.h_bss->KernelRadius, 0.1, 0.1, 20.0);
			sph_changed |= ImGui::DragFloat("Viscosity", &psys.h_bss->Viscosity, 0.1, 0.1, 100.0);
			sph_changed |= ImGui::DragFloat("Rho 0",   &psys.h_bss->RestDensity, 0.1, 0.1, 100.0);
			sph_changed |= ImGui::DragFloat("Gas constant (k)", &psys.h_bss->GasConst, 0.1, 0.1, 100.0);
			sph_changed |= ImGui::DragFloat("Particle mass", &psys.h_bss->PartMass, 0.1, 0.1, 100.0);
			sph_changed |= ImGui::DragFloat("Col restitution", &psys.h_bss->ColRestitution, 0.1, 0.1, 100.0);

			ImGui::End();

			ImGui::Render();

			if (sph_changed) {
				cudaMemcpy(psys.d_bss, psys.h_bss, sizeof(sph_sim_settings), cudaMemcpyHostToDevice);
			}

			LOG_TIMING("ImGui tab setting: {} ms", timer.swap_time());

			float dt = dtTimer.swap_time(); // dt is in ms
			for (int c = 0; c < sim_frames_per_frame && run_simulation; c++) {
				psys.update(dt / 10000.0);
				//printf("timestep\n");
			}

			LOG_TIMING("Particle update time: {} ms", timer.swap_time());

			// clear
			glClear(GL_COLOR_BUFFER_BIT);
			glClearColor(0.7, 0.7, 0.7, 0.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			GLCHECKERR();

			psr.renderps(m_width,m_height);
			GLCHECKERR();

			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			LOG_TIMING("Render time: {} ms", timer.swap_time());

			glfwSwapBuffers(window);
			glfwPollEvents();

			dispatch_events();

			LOG_TIMING("Event handling + swaping buffers time: {} ms", timer.swap_time());

		}
	}
};


int main(){
	MyApp app(1200,800,"Main Example");
	app.run();
}