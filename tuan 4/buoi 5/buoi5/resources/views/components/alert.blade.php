@if(session('success'))
  <div class="alert alert-success m-2">
    <script>
      alert('Thành Công')
    </script>
    {{ session('success') }}
  </div>
@endif
