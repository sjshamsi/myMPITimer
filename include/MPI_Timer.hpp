#pragma once

#include <mpi.h>
#include <chrono>
#include <iostream>
#include <string_view>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <type_traits>

namespace MPITiming
{

constexpr int DEFAULT_ROOT {0};



// ------------------------------
// Common types & settings which are overridable after the import
// ------------------------------
using clock_t = std::chrono::steady_clock;
using time_t = std::chrono::time_point<clock_t>;
using delta_t = clock_t::duration;


// ------------------------------
// Some print helpers
// ------------------------------
template <class T> struct is_chrono_duration : std::false_type {};
template <class Rep, class Period> struct is_chrono_duration<std::chrono::duration<Rep, Period>> : std::true_type {};

template <class To, class Rep, class Period>
constexpr double to_unit_count(const std::chrono::duration<Rep, Period>& d) {
  static_assert(is_chrono_duration<To>::value, "to_unit_count: `To` must be a std::chrono::duration type");

  using PeriodTag = typename To::period;
  using FloatTarget = std::chrono::duration<double, PeriodTag>;

  return std::chrono::duration_cast<FloatTarget>(d).count();
}


template <class TU>
constexpr const char* unit_name() {
  static_assert(is_chrono_duration<TU>::value, "unit_name: `TU` must be a std::chrono::duration type");
  
  using P = typename TU::period;

  if constexpr (std::is_same_v<P, std::nano>) return "ns";
  else if constexpr (std::is_same_v<P, std::micro>) return "µs";
  else if constexpr (std::is_same_v<P, std::milli>) return "ms";
  else if constexpr (std::is_same_v<P, std::ratio<1>>) return "s";
  else if constexpr (std::is_same_v<P, std::ratio<60>>) return "min";
  else if constexpr (std::is_same_v<P, std::ratio<3600>>) return "h";
  else return "ticks";
}


// ------------------------------
// STRUCT to store times in a thread
// ------------------------------
struct Store {
  uint32_t ncalls {};
  delta_t total_duration {};

  Store() = default;
  Store(const delta_t elapsed) : ncalls(1), total_duration(elapsed) {}
  Store(uint32_t number_calls, delta_t elapsed) : ncalls(number_calls), total_duration(elapsed) {}
  void add(delta_t elapsed) { ncalls++; total_duration += elapsed; }
};


// ------------------------------
// STRUCT to transport durations with MPI
// ------------------------------
template <typename TU>
struct Packet {
  const uint32_t ncalls;
  const double total_duration;
  const double avg_duration;

  Packet() : ncalls(uint32_t{}), total_duration(double{}), avg_duration(double{}) {}

  explicit Packet(const uint32_t number_calls, const delta_t elapsed) :
    ncalls(number_calls),
    total_duration(to_unit_count<TU>(elapsed)),
    avg_duration(ncalls ? total_duration / ncalls : double{}) {}

  explicit Packet(const Store &s) :
    ncalls(s.ncalls),
    total_duration(to_unit_count<TU>(s.total_duration)),
    avg_duration(ncalls ? total_duration / ncalls : double{}) {}
};


// ------------------------------
// CLASS Timepeeker
// ------------------------------
class Timekeeper {
public:
  friend class ScopedTimer;
  friend class Stopwatch;
  explicit Timekeeper(const MPI_Comm mpi_comm = MPI_COMM_WORLD, const int root_rank = DEFAULT_ROOT);
  
  template <typename TU>
  Packet<TU> get_packet(std::string_view subject);

  template <typename TU>
  std::ostream & packet_header(std::ostream& os = std::cout, std::string_view delim = ", ") const;

  template <typename TU>
  std::ostream & print_subject_packets(std::string_view subject, std::ostream& os = std::cout, std::string_view delim = ", ");

  template <typename TU>
  std::ostream & print_all_packets (std::ostream& os = std::cout, std::string_view delim = ", ");

  template <typename TU>
  std::ostream& summaries_header(std::ostream& os = std::cout, std::string_view delim = ", ") const;

  template <typename TU>
  std::ostream& print_subject_summary(std::string_view subject, std::ostream& os = std::cout, std::string_view delim = ", ");

  template <typename TU>
  std::ostream& print_all_summaries(std::ostream& os = std::cout, std::string_view delim = ", ");

private:
  MPI_Comm comm;
  int root, rank, csize;
  std::map<std::string, Store, std::less<>> all_timings;
  void add_timing(const delta_t elapsed, std::string_view subject);
};


Timekeeper::Timekeeper(const MPI_Comm mpi_comm, const int root_rank) :
  comm(mpi_comm), root(root_rank) {
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &csize);
}


void Timekeeper::add_timing(const delta_t elapsed, std::string_view subject) {
  auto it = all_timings.find(subject);
  if (it != all_timings.end()) {
    it->second.add(elapsed);
  } else {
    all_timings.emplace(subject, Store(elapsed));
  }
}


template <typename TU>
Packet<TU> Timekeeper::get_packet(std::string_view subject) {
  static_assert(std::is_trivially_copyable_v<Packet<TU>>, "Packet must be trivially copyable");

  auto it = all_timings.find(subject);
  if (it != all_timings.end()) {
    return Packet<TU>(it->second);
  } else {
    auto [new_it, inserted] = all_timings.emplace(subject, Store{});
    return Packet<TU>(new_it->second);
  }
}


template <typename TU>
std::ostream & Timekeeper::packet_header(std::ostream& os, std::string_view delim) const {
  if (rank != root) return os;
  
  static constexpr const char* unit = unit_name<TU>();
  os  << "rank"                       << delim
      << "subject"                    << delim
      << "tot_time["  << unit << "]"  << delim
      << "ncalls"                     << delim
      << "avg_time["  << unit << "]";
  return os;
}


template <typename TU>
std::ostream & Timekeeper::print_subject_packets(std::string_view subject, std::ostream& os, std::string_view delim) {
  Packet local_pkt = get_packet<TU>(subject);
  std::vector<Packet<TU>> all_packets;

  if (rank == root) all_packets.resize(csize);
  MPI_Gather(&local_pkt, static_cast<int>(sizeof(Packet<TU>)), MPI_BYTE,
             all_packets.data(), static_cast<int>(sizeof(Packet<TU>)), MPI_BYTE, 0, comm);
  if (rank != root) return os;

  for (int i = 0; i < csize; i++) {
    os << i                             << delim
       << subject                       << delim
       << all_packets[i].total_duration << delim
       << all_packets[i].ncalls         << delim
       << all_packets[i].avg_duration   << "\n";
  }
  return os;
}


template <typename TU>
std::ostream & Timekeeper::print_all_packets (std::ostream& os, std::string_view delim) {
  for (const auto& [subject, pkt] : all_timings) {
    print_subject_packets<TU>(subject, os, delim);
  }
  return os;
}


template <typename TU>
std::ostream& Timekeeper::summaries_header(std::ostream& os, std::string_view delim) const {
  if (rank != root) return os;
  
  static constexpr const char* unit = unit_name<TU>();
  os << "nprocs" << delim;

  for (const auto& [subject, pkt] : all_timings) {
    os << subject << "_tot_mean[" << unit << "]" << delim
       << subject << "_tot_min["  << unit << "]" << delim
       << subject << "_tot_max["  << unit << "]" << delim
       << subject << "_tot_stdv[" << unit << "]" << delim
       << subject << "_avg_mean[" << unit << "]" << delim
       << subject << "_avg_min["  << unit << "]" << delim
       << subject << "_avg_max["  << unit << "]" << delim
       << subject << "_avg_stdv[" << unit << "]";
  }
  return os;
}


template <typename TU>
std::ostream & Timekeeper::print_subject_summary(std::string_view subject, std::ostream& os, std::string_view delim) {
  Packet local_pkt = get_packet<TU>(subject);
  std::vector<Packet<TU>> all_packets;

  if (rank == root) all_packets.resize(csize);
  MPI_Gather(&local_pkt, static_cast<int>(sizeof(Packet<TU>)), MPI_BYTE,
             all_packets.data(), static_cast<int>(sizeof(Packet<TU>)), MPI_BYTE, 0, comm);
  if (rank != root) return os;

  double tot_mean{}, avg_mean{};
  int n_nonzero{};
  std::vector<double> tot_durations, avg_durations;
  tot_durations.reserve(csize), avg_durations.reserve(csize);

  for (int i = 0; i < csize; i++) {
    if (all_packets[i].ncalls) {
      n_nonzero ++;

      tot_durations.push_back(all_packets[i].total_duration);
      avg_durations.push_back(all_packets[i].avg_duration);

      tot_mean += all_packets[i].total_duration;
      avg_mean += all_packets[i].avg_duration;
    }
  }

  tot_mean /= n_nonzero;
  avg_mean /= n_nonzero;
  const auto [tot_min_it, tot_max_it] = std::minmax_element(tot_durations.begin(), tot_durations.end());
  const auto [avg_min_it, avg_max_it] = std::minmax_element(avg_durations.begin(), avg_durations.end());

  const double tot_min{*tot_min_it}, tot_max{*tot_max_it}, avg_min{*avg_min_it}, avg_max{*avg_max_it};

  const double tot_sq_sum = std::inner_product(tot_durations.begin(), tot_durations.end(), tot_durations.begin(), 0.0);
  const double tot_variance = tot_sq_sum / tot_durations.size() - (tot_mean * tot_mean);
  const double tot_stdv = std::sqrt(tot_variance > 0 ? tot_variance : 0.0);

  const double avg_sq_sum = std::inner_product(avg_durations.begin(), avg_durations.end(), avg_durations.begin(), 0.0);
  const double avg_variance = avg_sq_sum / avg_durations.size() - (avg_mean * avg_mean);
  const double avg_stdv = std::sqrt(avg_variance > 0 ? avg_variance : 0.0);
  
  os << tot_mean  << delim
     << tot_min   << delim
     << tot_max   << delim
     << tot_stdv  << delim
     << avg_mean  << delim
     << avg_min   << delim
     << avg_max   << delim
     << avg_stdv;

  return os;
}


template <typename TU>
std::ostream & Timekeeper::print_all_summaries(std::ostream& os, std::string_view delim) {
  if (rank == root) os << csize;
  for (const auto& [subject, pkt] : all_timings) {
    if (rank == root) os << delim;
    print_subject_summary<TU>(subject, os, delim);
  }
  if (rank == root) os << "\n";
  return os;
}


// ------------------------------
// CLASS ScopedTimer to do the timing
// ------------------------------
class ScopedTimer {
public:
  explicit ScopedTimer(Timekeeper& timekeeper, std::string_view timing_subject);
  ~ScopedTimer();

private:
  time_t start;
  std::string_view subject;
  Timekeeper& tk;
};

ScopedTimer::ScopedTimer(Timekeeper& timekeeper, std::string_view timing_subject) :
  start(clock_t::now()), subject(timing_subject), tk(timekeeper) {}

ScopedTimer::~ScopedTimer() {
  time_t end = clock_t::now();
  tk.add_timing(end - start, subject);
}


// ------------------------------
// CLASS Stopwatch to do the timing
// ------------------------------
class Stopwatch {
public:
  explicit Stopwatch(Timekeeper& timekeeper) : tk(timekeeper) {}
  void start(std::string_view timing_subject);
  void stop();

private:
  Timekeeper& tk;
  time_t start_time;
  std::string_view subject;
  bool in_progress {false};
};


void Stopwatch::start(std::string_view timing_subject) {
  start_time = clock_t::now();

  if (in_progress) {
    if (tk.rank == tk.root) std::cerr << "Timing already in progress/done for this timer! Use a new instance?\n";
    MPI_Abort(tk.comm, 1);
  }

  in_progress = true;
  subject = timing_subject;
}


void Stopwatch::stop() {
  time_t end_time = clock_t::now();

  if (!in_progress) {
    if (tk.rank == tk.root) std::cerr << "No timing in progress for this timer!\n";
    MPI_Abort(tk.comm, 1);
  }

  tk.add_timing(end_time - start_time, subject);
}

} // namespace MPITiming