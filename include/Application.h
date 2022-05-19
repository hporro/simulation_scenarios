#pragma once

#include <GL/glew.h>

#include "imgui.h"
#include "lib/imgui/backends/imgui_impl_glfw.h"
#include "lib/imgui/backends/imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <vector>

#include "Logging.h"
#include "glErrCheck.h"
#include "gpuErrCheck.h"
#include "timing_helpers.h"

enum Event_IDs {
	Null = 0,
	WindowResize = 1,
	Button = 2,
};

struct Event { Event_IDs m_id; bool handled = false; Event(Event_IDs evid) : m_id(evid) {} };

struct WindowResizeEvent : public Event {
	WindowResizeEvent(int width, int height) : m_width(width), m_height(height), Event(WindowResize) {}
	int m_width, m_height;
};

struct ButtonEvent : public Event {
	ButtonEvent(int key, int scancode, int action, int mods) : m_key(key), m_scancode(scancode), m_action(action), m_mods(mods), Event(Button) {}
	int m_key, m_scancode, m_action, m_mods;
};

std::vector<Event*> eventQueue;

struct Application {
	Application(int width, int height, const std::string title) : m_width(width), m_height(height) {
		Timer timer;

		if (!logging_initialized) {
			init_logging();
			LOG_EVENT("Logging initialized");
		}

		if (!opengl_context_initialized) {

			// initialize the GLFW library
			if (!glfwInit()) {
				LOG_EVENT("Couldn't init GLFW");
				throw std::runtime_error("Couldn't init GLFW");
			}
			LOG_EVENT("GLFW initialized");

			// setting the opengl version
			int major = 3;
			int minor = 2;
			glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
			glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
			glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
			glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

			// create the window
			window = glfwCreateWindow(m_width, m_height, title.c_str(), NULL, NULL);
			if (!window) {
				glfwTerminate();
				LOG_EVENT("Couldn't create a GLFW window");
				throw std::runtime_error("Couldn't create a window");
			}

			glfwMakeContextCurrent(window);
			setup_callbacks();
			
			glewExperimental = GL_TRUE;
			GLenum err = glewInit();

			if (err != GLEW_OK) {
				glfwTerminate();
				LOG_EVENT(std::string("Could initialize GLEW, error = ") +
					(const char*)glewGetErrorString(err));
				throw std::runtime_error(std::string("Could initialize GLEW, error = ") +
					(const char*)glewGetErrorString(err));
			}

			// get version info
			const GLubyte* renderer = glGetString(GL_RENDERER);
			const GLubyte* version = glGetString(GL_VERSION);
			
			LOG_EVENT("Renderer: {}", renderer);
			LOG_EVENT("OpenGL version supported {}", version);

			// opengl configuration
			glEnable(GL_DEPTH_TEST);  // enable depth-testing
			glDepthFunc(GL_LESS);  // depth-testing interprets a smaller value as "closer"

			glfwWindowHint(GLFW_SAMPLES, 4);
			glEnable(GL_MULTISAMPLE);
			GLCHECKERR();

			// vsync
			// glfwSwapInterval(false);
			opengl_context_initialized = true;
			LOG_EVENT("OpenGL context initialized");
		}
		
		if (opengl_context_initialized && !imgui_context_initialized) {
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
			//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

			// Setup Dear ImGui style
			ImGui::StyleColorsDark();
			//ImGui::StyleColorsClassic();

			// Setup Platform/Renderer backends
			ImGui_ImplGlfw_InitForOpenGL(window, true);
			ImGui_ImplOpenGL3_Init("#version 130");
			imgui_context_initialized = true;

			LOG_EVENT("ImGui context initialized");
		}

		LOG_TIMING("Application base class initialization time: {} ms", timer.get_time());
	}

	void run() {

		while (!glfwWindowShouldClose(window)) {

			Timer timer;

			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::SetNextWindowSize(ImVec2(m_width/4, m_height)); // ensures ImGui fits the GLFW window
			ImGui::SetNextWindowPos(ImVec2(0, 0));

			ImGui::Begin("Simulation info/settings", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
			ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::End();

			ImGui::Render();

			LOG_TIMING("ImGui tab setting: {}", timer.swap_time());

			// clear
			glClear(GL_COLOR_BUFFER_BIT);
			glClearColor(0.0, 0.0, 0.0, 0.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			GLCHECKERR();

			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			LOG_TIMING("Render time: {}", timer.swap_time());


			glfwSwapBuffers(window);
			glfwPollEvents();

			dispatch_events();

			LOG_TIMING("Event handling time: {}", timer.swap_time());

		}

	}

	GLFWwindow* window = nullptr;
	bool opengl_context_initialized = false;
	bool logging_initialized = false;
	bool imgui_context_initialized = false;
	int m_width = 0, m_height = 0;

	void setWidthHeight(int width, int height) {
		m_width = width;
		m_height = height;
	}

	void setup_callbacks() {
		glfwSetKeyCallback(window, 
			[](GLFWwindow* window, int key, int scancode, int action, int mods) {
				eventQueue.push_back(new ButtonEvent(key, scancode, action, mods));
			}
		);
		glfwSetWindowSizeCallback(window, 
			[](GLFWwindow * window, int width, int height) {
				eventQueue.push_back(new WindowResizeEvent{ width, height });
		});
		LOG_EVENT("Callbacks setup");
	}

	void dispatch_events() {
		for (Event* e : eventQueue) {
			switch (e->m_id) {
			case WindowResize: {
				WindowResizeEvent* wre = static_cast<WindowResizeEvent*>(e);
				setWidthHeight(wre->m_width, wre->m_height);
				glViewport(0, 0, m_width, m_height);
				// TODO: do some other stuff maybe
				wre->handled = true;
			}
			break;
			case Button: {
				ButtonEvent* be = static_cast<ButtonEvent*>(e);
				if (be->m_key == GLFW_KEY_ESCAPE && be->m_action == GLFW_PRESS) {
					glfwSetWindowShouldClose(window, GLFW_TRUE);
					be->handled = true;
				}
			}
			break;
			default:
			break;
			}
		}
		eventQueue.clear();
	}
};