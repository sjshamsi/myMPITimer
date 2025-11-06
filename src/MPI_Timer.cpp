#include <array>
#include <thread>

#include "MPI_Timer.hpp"

using TimeUnits = std::chrono::microseconds;


constexpr std::array<std::string_view, 4> jobs = {"Job1", "Job2", "Job3", "Job4"};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    if (rank == 0) std::cerr << "I need at least 2 processes!\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPITiming::Timekeeper tk;
  
  MPITiming::Stopwatch sw0(tk);
  sw0.start("Eating");
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  sw0.stop();

  MPITiming::Stopwatch sw1(tk);
  sw1.start("Eating");
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  sw1.stop();

  MPITiming::Stopwatch sw2(tk);
  sw2.start("Eating");
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  sw2.stop();

  // for (auto &job : jobs) {
  //   MPITiming::Stopwatch sw(tk);
  //   sw.start(job);
  //   std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  //   sw.stop();
  // }
  tk.packet_header<TimeUnits>();
  if (rank == 0) std::cout << "\n";
  tk.print_all_packets<TimeUnits>();

  if (rank == 0) std::cout << "\n";
  tk.summaries_header<TimeUnits>();
  if (rank == 0) std::cout << "\n";
  tk.print_all_summaries<TimeUnits>();

  MPI_Finalize();
  return 0;
}